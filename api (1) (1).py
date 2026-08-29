"""
serving/api.py
─────────────────────────────────────────────────────────────────────────────
Production-ready FastAPI inference service — §6

Implements:
  - Full inference pipeline: feature fetch → Stage 1 retrieval → Stage 2 ranking
    → post-ranking (MMR + fairness) → response
  - End-to-end latency budget: < 200ms (target), < 300ms (hard limit)
  - Circuit breaker: falls back to popularity baseline if latency > 250ms (§6.4)
  - Request coalescing: batches calls from same session within 500ms (§6.3)

Run with:
  uvicorn serving.api:app --host 0.0.0.0 --port 8080 --workers 4
"""

from __future__ import annotations

import time
import asyncio
import numpy as np
import torch
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from config import CFG, PostRankingConfig
from features.pipeline import (
    FeatureStore, build_context_features, compute_meal_completeness,
    compute_cross_features, ComplementarityMatrix, get_mealtime,
    MealTimeSparsityFallback,
)
from models.two_tower import TwoTowerModel, FAISSRetrievalIndex
from models.dctn import DCTN
from models.post_ranking import PostRankingPipeline, ScoredCandidate


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class CartItem(BaseModel):
    item_id: str
    category: str
    cuisine: str
    price: float

class RecommendationRequest(BaseModel):
    session_id: str
    user_id: str
    restaurant_id: str
    cart_items: List[CartItem]              # Current cart state
    cuisine_type: str = "default"
    city: str = "Mumbai"
    timestamp: Optional[str] = None        # ISO format; defaults to now

class RecommendationItem(BaseModel):
    rank: int
    item_id: str
    p_accept: float
    aov_lift_inr: float
    final_score: float
    category: str
    is_chain: bool
    restaurant_id: str

class RecommendationResponse(BaseModel):
    session_id: str
    recommendations: List[RecommendationItem]
    served_by: str          # "model" or "fallback"
    latency_ms: float
    top_n: int


# ─────────────────────────────────────────────────────────────────────────────
# POPULARITY BASELINE  (circuit breaker fallback — §6.4)
# ─────────────────────────────────────────────────────────────────────────────

class PopularityBaseline:
    """
    Fallback recommendation: co-purchase frequency heuristic.
    This is also the A/B test control arm (§7.1).

    Ranks add-ons by raw co-occurrence count within the same restaurant
    over the last 30 days. No user personalisation or sequential modelling.
    """

    def __init__(self, popularity_data: Dict[str, List[str]]):
        """popularity_data: {restaurant_id: [item_id in popularity order]}"""
        self.data = popularity_data

    def recommend(
        self,
        restaurant_id: str,
        cart_item_ids: List[str],
        top_n: int = 8,
    ) -> List[Dict]:
        popular = self.data.get(restaurant_id, [])
        cart_set = set(cart_item_ids)
        filtered = [item_id for item_id in popular if item_id not in cart_set]

        return [
            {
                "rank": i + 1,
                "item_id": iid,
                "p_accept": max(0.1, 0.35 - i * 0.02),
                "aov_lift_inr": 0.0,
                "final_score": max(0.1, 1.0 - i * 0.1),
                "category": "unknown",
                "is_chain": True,
                "restaurant_id": restaurant_id,
            }
            for i, iid in enumerate(filtered[:top_n])
        ]


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Orchestrates the full inference pipeline with latency tracking.
    Implements §6.1 latency budget breakdown.
    """

    def __init__(
        self,
        two_tower: TwoTowerModel,
        faiss_index: FAISSRetrievalIndex,
        dctn: DCTN,
        feature_store: FeatureStore,
        post_ranking: PostRankingPipeline,
        baseline: PopularityBaseline,
        item_metadata: Dict[str, Dict],     # {item_id: {category, cuisine, price, is_chain, restaurant_id}}
        item_embeddings: Dict[str, np.ndarray],
        comp_matrix: ComplementarityMatrix,
        device: str = "cpu",
        circuit_breaker_ms: float = 250.0,
    ):
        self.two_tower        = two_tower
        self.faiss_index      = faiss_index
        self.dctn             = dctn
        self.feature_store    = feature_store
        self.post_ranking     = post_ranking
        self.baseline         = baseline
        self.item_metadata    = item_metadata
        self.item_embeddings  = item_embeddings
        self.comp_matrix      = comp_matrix
        self.device           = device
        self.circuit_breaker_ms = circuit_breaker_ms

        self.two_tower.eval()
        self.dctn.eval()

    async def predict(self, request: RecommendationRequest) -> RecommendationResponse:
        t0 = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self._predict_with_model(request),
                timeout=self.circuit_breaker_ms / 1000.0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            result.latency_ms = round(latency_ms, 2)
            return result

        except asyncio.TimeoutError:
            # Circuit breaker: fall back to popularity baseline
            latency_ms = (time.perf_counter() - t0) * 1000
            return self._fallback_response(request, latency_ms)

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"[InferenceEngine] Error: {e}. Falling back to baseline.")
            return self._fallback_response(request, latency_ms)

    async def _predict_with_model(
        self, request: RecommendationRequest
    ) -> RecommendationResponse:
        ts = (datetime.fromisoformat(request.timestamp)
              if request.timestamp else datetime.now())

        # ── Step 1: Feature retrieval (~15ms) ──────────────────────────────
        user_feat = self.feature_store.get("user", request.user_id)
        if user_feat is None:
            user_feat = np.zeros(32, dtype=np.float32)

        ctx_feat = build_context_features(ts, request.city)

        # ── Step 2: Cart analysis ───────────────────────────────────────────
        cart_item_ids = [c.item_id for c in request.cart_items]
        cart_cats     = [c.category for c in request.cart_items]
        cart_prices   = [c.price for c in request.cart_items]
        cart_cuisine  = (request.cart_items[0].cuisine
                         if request.cart_items else "default")

        completeness_score, missing_components = compute_meal_completeness(
            cart_cats, request.cuisine_type
        )

        # ── Step 3: Stage 1 — FAISS ANN retrieval (~20ms) ──────────────────
        query_vec = np.concatenate([user_feat[:64], user_feat[:64]])  # (128,)
        with torch.no_grad():
            q_tensor = torch.tensor(query_vec, dtype=torch.float).unsqueeze(0)
            q_embed  = self.two_tower.encode_query(q_tensor).cpu().numpy()[0]

        # Get restaurant menu items to constrain retrieval
        restaurant_items = [
            iid for iid, meta in self.item_metadata.items()
            if meta.get("restaurant_id") == request.restaurant_id
        ]
        candidate_ids = self.faiss_index.search(
            q_embed,
            top_k=CFG.two_tower.top_k_retrieval,
            restaurant_item_ids=restaurant_items,
        )

        # Exclude already-in-cart items
        cart_set      = set(cart_item_ids)
        candidate_ids = [c for c in candidate_ids if c not in cart_set]

        if not candidate_ids:
            return self._fallback_response(request, latency_ms=0)

        # ── Step 4: Stage 2 — DCTN ranking (~60ms) ─────────────────────────
        scored_candidates = await self._dctn_score_candidates(
            candidate_ids, request, user_feat, ctx_feat, cart_cats,
            cart_prices, cart_cuisine, missing_components, completeness_score,
        )

        # ── Step 5: Post-ranking (~5ms) ────────────────────────────────────
        final_list = self.post_ranking.run(scored_candidates)
        recs       = self.post_ranking.get_recommendation_response(final_list)

        return RecommendationResponse(
            session_id=request.session_id,
            recommendations=[RecommendationItem(**r) for r in recs],
            served_by="model",
            latency_ms=0.0,   # filled in by caller
            top_n=len(recs),
        )

    async def _dctn_score_candidates(
        self,
        candidate_ids: List[str],
        request: RecommendationRequest,
        user_feat: np.ndarray,
        ctx_feat: np.ndarray,
        cart_cats: List[str],
        cart_prices: List[float],
        cart_cuisine: str,
        missing_components: List[str],
        completeness_score: float,
    ) -> List[ScoredCandidate]:
        """Batch DCTN inference over all candidates."""
        B = len(candidate_ids)
        max_cart = CFG.dctn.cart_encoder.max_cart_len

        # Map item IDs to integer indices (vocab lookup)
        item_to_idx = {iid: i + 1 for i, iid in enumerate(self.item_metadata.keys())}

        # Build cart_ids tensor (same for all candidates)
        cart_indices = [item_to_idx.get(iid, 0) for iid in
                        request.cart_items[-max_cart:]]
        pad_len      = max_cart - len(cart_indices)
        cart_padded  = [0] * pad_len + cart_indices
        attn_mask    = [0] * pad_len + [1] * len(cart_indices)

        cart_tensor  = torch.tensor([cart_padded] * B, dtype=torch.long)
        attn_tensor  = torch.tensor([attn_mask] * B, dtype=torch.long)
        user_tensor  = torch.tensor([user_feat] * B, dtype=torch.float)
        ctx_tensor   = torch.tensor([ctx_feat] * B, dtype=torch.float)
        comp_tensor  = torch.tensor([[completeness_score]] * B, dtype=torch.float)

        # Build per-candidate features
        item_feats_list  = []
        cross_feats_list = []

        for cand_id in candidate_ids:
            meta       = self.item_metadata.get(cand_id, {})
            item_feat  = self.feature_store.get("item", cand_id)
            if item_feat is None:
                item_feat = np.zeros(32, dtype=np.float32)

            comp_scores = self.comp_matrix.get_candidate_scores(
                meta.get("category", "main"), cart_cats
            )
            cross_feat = compute_cross_features(
                candidate_price=meta.get("price", 100.0),
                candidate_category=meta.get("category", "main"),
                candidate_cuisine=meta.get("cuisine", "default"),
                cart_prices=cart_prices,
                cart_categories=cart_cats,
                cart_dominant_cuisine=cart_cuisine,
                missing_components=missing_components,
                complementarity_scores=comp_scores,
            )
            item_feats_list.append(item_feat)
            cross_feats_list.append(cross_feat)

        item_tensor  = torch.tensor(item_feats_list, dtype=torch.float)
        cross_tensor = torch.tensor(cross_feats_list, dtype=torch.float)

        with torch.no_grad():
            p_accept, aov_lift, final_score, _ = self.dctn(
                cart_ids=cart_tensor,
                attention_mask=attn_tensor,
                user_features=user_tensor,
                item_features=item_tensor,
                context_features=ctx_tensor,
                cross_features=cross_tensor,
                completeness_score=comp_tensor,
                apply_masking=False,
            )

        p_accept_np   = p_accept.cpu().numpy()
        aov_lift_np   = aov_lift.cpu().numpy()
        final_score_np = final_score.cpu().numpy()

        scored = []
        for i, cand_id in enumerate(candidate_ids):
            meta = self.item_metadata.get(cand_id, {})
            emb  = self.item_embeddings.get(cand_id)
            scored.append(ScoredCandidate(
                item_id=cand_id,
                relevance_score=float(final_score_np[i]),
                p_accept=float(p_accept_np[i]),
                aov_lift=float(aov_lift_np[i]),
                category=meta.get("category", "unknown"),
                cuisine=meta.get("cuisine", "unknown"),
                price=float(meta.get("price", 0.0)),
                is_chain=bool(meta.get("is_chain", True)),
                restaurant_id=meta.get("restaurant_id", request.restaurant_id),
                item_embedding=emb,
            ))

        return scored

    def _fallback_response(
        self, request: RecommendationRequest, latency_ms: float
    ) -> RecommendationResponse:
        """Popularity-based fallback (circuit breaker or error path)."""
        cart_ids = [c.item_id for c in request.cart_items]
        recs     = self.baseline.recommend(request.restaurant_id, cart_ids)
        return RecommendationResponse(
            session_id=request.session_id,
            recommendations=[RecommendationItem(**r) for r in recs],
            served_by="fallback",
            latency_ms=round(latency_ms, 2),
            top_n=len(recs),
        )


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CSAO Rail Recommendation API",
    description="Zomathon 2025 — Cart Super Add-On Recommendation System",
    version="2.0.0",
)

# Engine is initialised at startup (loaded from disk or passed in)
engine: Optional[InferenceEngine] = None


@app.on_event("startup")
async def startup_event():
    """
    In production: load models from disk, build FAISS index, connect to Redis.
    Here we log readiness.
    """
    print("[API] CSAO Rail service started. Waiting for engine initialisation.")


@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": engine is not None}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Main recommendation endpoint.
    Target latency: < 200ms (hard limit: 300ms).
    Falls back to popularity baseline if model exceeds circuit breaker threshold.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialised.")

    response = await engine.predict(request)

    # Log for monitoring
    status = "OK" if response.latency_ms < 200 else "SLOW"
    print(f"[{status}] session={request.session_id} "
          f"latency={response.latency_ms:.1f}ms "
          f"served_by={response.served_by} "
          f"n={response.top_n}")

    return response


@app.post("/recommend/batch")
async def recommend_batch(requests: List[RecommendationRequest]):
    """
    Request coalescing endpoint (§6.3):
    Multiple add-to-cart events from the same session batched into one call.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialised.")

    # Group by session and take the latest cart state
    session_map: Dict[str, RecommendationRequest] = {}
    for req in requests:
        session_map[req.session_id] = req   # Last write wins (latest cart state)

    responses = await asyncio.gather(*[
        engine.predict(req) for req in session_map.values()
    ])
    return responses
