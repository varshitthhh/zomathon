"""
models/post_ranking.py
─────────────────────────────────────────────────────────────────────────────
Post-Ranking Layer — §4.4

Two components applied after DCTN scoring:
  1. MMR Re-Ranker    — balances relevance vs. diversity
  2. Fairness Floor   — guarantees independent restaurant exposure

These run on the final ranked list (~5ms) before serving.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from config import PostRankingConfig


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE ITEM  (shared data structure)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredCandidate:
    item_id: str
    relevance_score: float          # From DCTN final_score
    p_accept: float
    aov_lift: float
    category: str
    cuisine: str
    price: float
    is_chain: bool
    restaurant_id: str
    item_embedding: Optional[np.ndarray] = None   # 128-dim for MMR similarity


# ─────────────────────────────────────────────────────────────────────────────
# 1. MMR RE-RANKER  (§4.4 — New Addition)
# ─────────────────────────────────────────────────────────────────────────────

class MMRReRanker:
    """
    Maximal Marginal Relevance (MMR) re-ranker.

    Score_MMR = λ × relevance_score − (1−λ) × max_similarity_to_selected

    Prevents the rail from showing near-identical items (e.g., 5 biryani
    variants). Each slot adds incremental value to the recommendation set.

    λ = 0.7 (default): weights relevance 70%, diversity 30%.
    """

    def __init__(self, lambda_mmr: float = 0.7):
        self.lambda_mmr = lambda_mmr

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / norm)

    def _category_sim(self, cat_a: str, cat_b: str) -> float:
        """Fallback similarity when embeddings are unavailable."""
        return 1.0 if cat_a == cat_b else 0.0

    def rerank(
        self,
        candidates: List[ScoredCandidate],
        top_n: int = 8,
    ) -> List[ScoredCandidate]:
        """
        Greedily select top_n items using MMR.
        Candidates must already be sorted by relevance_score (descending).
        """
        if len(candidates) <= top_n:
            return candidates

        selected: List[ScoredCandidate] = []
        remaining = list(candidates)

        while len(selected) < top_n and remaining:
            best_item  = None
            best_score = -float("inf")

            for candidate in remaining:
                relevance = candidate.relevance_score

                if not selected:
                    mmr_score = relevance
                else:
                    # Max similarity to any already-selected item
                    if candidate.item_embedding is not None:
                        sims = [
                            self._cosine_sim(candidate.item_embedding, s.item_embedding)
                            for s in selected if s.item_embedding is not None
                        ]
                    else:
                        # Fallback: use category-level similarity
                        sims = [self._category_sim(candidate.category, s.category)
                                for s in selected]

                    max_sim   = max(sims) if sims else 0.0
                    mmr_score = (self.lambda_mmr * relevance
                                 - (1 - self.lambda_mmr) * max_sim)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item  = candidate

            if best_item is not None:
                selected.append(best_item)
                remaining.remove(best_item)

        return selected


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAIRNESS FLOOR  (§4.4 — New Addition)
# ─────────────────────────────────────────────────────────────────────────────

class FairnessFloor:
    """
    Guarantees minimum exposure for independent (non-chain) restaurants.

    Problem: Chain restaurants have richer interaction histories, causing the
    model to systematically suppress independent restaurants.

    Solution: If no independent restaurant item appears in top-N, insert at
    least one at position `insert_position` (default: 7 or 8).

    This aligns with the platform fairness goal of not marginalising
    small merchants solely due to data sparsity.
    """

    def __init__(self, min_independent: int = 1, insert_position: int = 7):
        self.min_independent = min_independent
        self.insert_position = insert_position   # 0-indexed

    def apply(
        self,
        ranked: List[ScoredCandidate],
        all_candidates: List[ScoredCandidate],
        top_n: int = 8,
    ) -> List[ScoredCandidate]:
        """
        Apply fairness floor to the final ranked list.
        Modifies the list in-place (replaces last position if needed).
        """
        # Count independent restaurant items in current top-N selection
        indie_count = sum(1 for c in ranked[:top_n] if not c.is_chain)

        if indie_count >= self.min_independent:
            return ranked[:top_n]

        # Find best independent candidate not already in the list
        selected_ids = {c.item_id for c in ranked[:top_n]}
        indie_candidates = [
            c for c in all_candidates
            if not c.is_chain and c.item_id not in selected_ids
        ]

        if not indie_candidates:
            return ranked[:top_n]   # No independent candidates available

        # Sort by relevance and pick the best
        indie_candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        best_indie = indie_candidates[0]

        # Insert at specified position (replace last item)
        result = list(ranked[:top_n])
        insert_at = min(self.insert_position, len(result) - 1)
        result[insert_at] = best_indie

        return result


# ─────────────────────────────────────────────────────────────────────────────
# POST-RANKING PIPELINE  (orchestrates MMR + Fairness)
# ─────────────────────────────────────────────────────────────────────────────

class PostRankingPipeline:
    """
    Full post-ranking pipeline. Applied after DCTN scoring.
    Estimated runtime: ~5ms for typical candidate lists (top-100 → top-8).
    """

    def __init__(self, cfg: PostRankingConfig):
        self.cfg      = cfg
        self.mmr      = MMRReRanker(lambda_mmr=cfg.mmr_lambda)
        self.fairness = FairnessFloor(
            min_independent=cfg.fairness_floor,
            insert_position=cfg.fairness_position,
        )

    def run(
        self,
        candidates: List[ScoredCandidate],  # Top-100 from Stage 1, scored by DCTN
        top_n: Optional[int] = None,
    ) -> List[ScoredCandidate]:
        """
        1. Sort by relevance score (descending)
        2. MMR re-ranking for diversity
        3. Fairness floor for independent restaurants
        4. Return top-N recommendations
        """
        n = top_n or self.cfg.top_n

        # Sort by DCTN final score
        candidates_sorted = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)

        # Step 1: MMR re-ranking
        mmr_ranked = self.mmr.rerank(candidates_sorted, top_n=n)

        # Step 2: Fairness floor
        final_list = self.fairness.apply(mmr_ranked, candidates_sorted, top_n=n)

        return final_list

    def get_recommendation_response(
        self,
        final_list: List[ScoredCandidate],
    ) -> List[Dict]:
        """
        Format final recommendations for API response.
        Matches expected output: ranked list with probability scores.
        """
        return [
            {
                "rank":             rank + 1,
                "item_id":          c.item_id,
                "p_accept":         round(c.p_accept, 4),
                "aov_lift_inr":     round(c.aov_lift, 2),
                "final_score":      round(c.relevance_score, 4),
                "category":         c.category,
                "is_chain":         c.is_chain,
                "restaurant_id":    c.restaurant_id,
            }
            for rank, c in enumerate(final_list)
        ]
