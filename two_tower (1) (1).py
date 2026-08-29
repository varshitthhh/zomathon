"""
models/two_tower.py
─────────────────────────────────────────────────────────────────────────────
Stage 1: Two-Tower Neural Network for Candidate Retrieval (§4.2)

Architecture:
  - Query Tower  → encodes (User, Cart, Context) → 128-dim query vector
  - Item Tower   → encodes each menu item        → 128-dim item vector
  - Trained with InfoNCE contrastive loss (in-batch negatives)
  - FAISS IVF-PQ index for sub-10ms ANN lookup
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict

from config import TwoTowerConfig


# ─────────────────────────────────────────────────────────────────────────────
# MLP TOWER  (shared building block)
# ─────────────────────────────────────────────────────────────────────────────

class MLPTower(nn.Module):
    """Generic MLP tower with layer normalisation and dropout."""

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev   = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)     # L2-normalise output


# ─────────────────────────────────────────────────────────────────────────────
# TWO-TOWER MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TwoTowerModel(nn.Module):
    """
    Two-Tower retrieval model.

    Query Tower: (user_features + context_features) → 128-dim
    Item Tower:  (item_features)                    → 128-dim

    Temperature τ is a *learnable* parameter (initialised at 0.07).
    """

    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg

        query_input_dim = cfg.user_feature_dim + cfg.item_feature_dim  # user + cart summary
        self.query_tower = MLPTower(
            input_dim=query_input_dim,
            hidden_dims=cfg.hidden_dims,
            output_dim=cfg.embedding_dim,
            dropout=cfg.dropout,
        )
        self.item_tower = MLPTower(
            input_dim=cfg.item_feature_dim,
            hidden_dims=cfg.hidden_dims,
            output_dim=cfg.embedding_dim,
            dropout=cfg.dropout,
        )
        # Learnable temperature (log-scale for numerical stability)
        self.log_temperature = nn.Parameter(torch.tensor(np.log(cfg.temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def encode_query(self, query_features: torch.Tensor) -> torch.Tensor:
        """Encode (user + cart summary) → L2-normalised 128-dim vector."""
        return self.query_tower(query_features)

    def encode_items(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features → L2-normalised 128-dim vectors."""
        return self.item_tower(item_features)

    def forward(
        self,
        query_features: torch.Tensor,        # (B, query_dim)
        pos_item_features: torch.Tensor,     # (B, item_dim)
    ) -> torch.Tensor:
        """Returns InfoNCE loss over the batch."""
        q = self.encode_query(query_features)    # (B, E)
        k = self.encode_items(pos_item_features) # (B, E)
        return infonce_loss(q, k, self.temperature)


# ─────────────────────────────────────────────────────────────────────────────
# INFONCE LOSS  (§4.2)
# ─────────────────────────────────────────────────────────────────────────────

def infonce_loss(
    queries: torch.Tensor,          # (B, E)
    keys: torch.Tensor,             # (B, E)
    temperature: torch.Tensor,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss with in-batch negatives.

    L = −log [ exp(q·k₊ / τ) / ∑ᴵ exp(q·kᴵ / τ) ]

    Every other item in the batch acts as a hard negative.
    Assumes queries and keys are already L2-normalised.
    """
    B = queries.shape[0]

    # (B, B) similarity matrix
    logits  = torch.matmul(queries, keys.T) / temperature   # (B, B)
    targets = torch.arange(B, device=queries.device)         # diagonal = positives
    loss    = F.cross_entropy(logits, targets)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# FAISS RETRIEVAL INDEX
# ─────────────────────────────────────────────────────────────────────────────

class FAISSRetrievalIndex:
    """
    Wraps a FAISS IVF-PQ index for fast approximate nearest-neighbour lookup.

    In production: sharded by city/restaurant and served in-memory on the
    inference pod. Rebuilt nightly; new items appended intraday via LLM
    embeddings (§4.2 FAISS staleness mitigation).
    """

    def __init__(self, embedding_dim: int = 128, nlist: int = 100, nprobe: int = 32):
        self.embedding_dim = embedding_dim
        self.nlist  = nlist
        self.nprobe = nprobe
        self._index = None
        self._item_ids: List[str] = []

    def _init_index(self):
        """Lazily import faiss and build an IVF-PQ index."""
        try:
            import faiss
            # IVF with flat quantiser — use flat if corpus < 10K items
            quantiser  = faiss.IndexFlatIP(self.embedding_dim)
            self._index = faiss.IndexIVFFlat(
                quantiser, self.embedding_dim, min(self.nlist, 10),
                faiss.METRIC_INNER_PRODUCT,
            )
            self._index.nprobe = self.nprobe
        except ImportError:
            print("[FAISSIndex] faiss not installed. Falling back to brute-force search.")
            self._index = None

    def build(self, item_ids: List[str], embeddings: np.ndarray):
        """
        Train and populate the index.
        embeddings: (N, embedding_dim), L2-normalised float32.
        """
        self._item_ids = item_ids
        self._embeddings = embeddings.astype(np.float32)

        self._init_index()
        if self._index is not None:
            self._index.train(self._embeddings)
            self._index.add(self._embeddings)
            print(f"[FAISSIndex] Built index with {len(item_ids):,} items.")
        else:
            print(f"[FAISSIndex] Brute-force index with {len(item_ids):,} items.")

    def append(self, new_item_ids: List[str], new_embeddings: np.ndarray):
        """Intraday append for new items (§4.2 staleness mitigation)."""
        self._item_ids.extend(new_item_ids)
        if hasattr(self, '_embeddings'):
            self._embeddings = np.vstack([self._embeddings,
                                          new_embeddings.astype(np.float32)])
        else:
            self._embeddings = new_embeddings.astype(np.float32)

        if self._index is not None:
            self._index.add(new_embeddings.astype(np.float32))
        print(f"[FAISSIndex] Appended {len(new_item_ids)} new items. "
              f"Total: {len(self._item_ids):,}")

    def search(
        self,
        query_embedding: np.ndarray,    # (embedding_dim,)
        top_k: int = 100,
        restaurant_item_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Return top-k item IDs closest to query.
        If restaurant_item_ids is provided, filter results to that restaurant.
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)

        if self._index is not None:
            _, indices = self._index.search(query, min(top_k * 3, len(self._item_ids)))
            candidates = [self._item_ids[i] for i in indices[0] if i >= 0]
        else:
            # Brute-force fallback
            sims      = self._embeddings @ query.T
            sorted_ix = np.argsort(-sims.flatten())
            candidates = [self._item_ids[i] for i in sorted_ix[:top_k * 3]]

        if restaurant_item_ids:
            restaurant_set = set(restaurant_item_ids)
            candidates = [c for c in candidates if c in restaurant_set]

        return candidates[:top_k]
