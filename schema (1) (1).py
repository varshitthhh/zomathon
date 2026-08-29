"""
data/schema.py
─────────────────────────────────────────────────────────────────────────────
Pydantic schemas for all data entities + PyTorch Dataset classes.
Assumes datasets are pre-loaded as pandas DataFrames matching the schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS  (mirrors document §2.1)
# ─────────────────────────────────────────────────────────────────────────────

class UserRecord(BaseModel):
    user_id: str
    city: str
    segment: str                        # budget / regular / premium
    signup_date: str
    preferred_cuisine: Optional[str]
    active_meal_times: List[str] = []   # e.g. ["dinner"] — sparse users

class RestaurantRecord(BaseModel):
    restaurant_id: str
    city: str
    cuisine_type: str
    price_range: str                    # low / mid / high
    chain_flag: bool
    rating: float = Field(ge=1.0, le=5.0)

class MenuItemRecord(BaseModel):
    item_id: str
    restaurant_id: str
    category: str                       # main / side / beverage / dessert / etc.
    sub_category: str
    price: float
    veg_flag: bool
    popularity_rank: int
    cuisine_type: str
    is_new: bool = False                # items added < 24h ago

class OrderSession(BaseModel):
    session_id: str
    user_id: str
    restaurant_id: str
    timestamp: str
    items_ordered: List[str]            # item_ids in order of addition
    items_shown_not_added: List[str]    # item_ids that were displayed but rejected
    meal_time: str
    city: str


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING EXAMPLE DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingExample:
    """
    One training example is a snapshot of the cart at step k:
      - cart_item_ids: sequence of items added so far (length k)
      - candidate_item_id: item being evaluated
      - label: 1 if item was eventually added, 0 if shown and rejected
      - user_features: pre-computed user feature vector
      - item_features: pre-computed candidate item feature vector
      - context_features: temporal + geo features
      - cross_features: candidate × cart interaction features
      - aov_lift: how much AOV increased when this item was added (0 for negatives)
      - is_chain: whether candidate restaurant is a chain
    """
    cart_item_ids: List[int]
    candidate_item_id: int
    label: int
    user_features: np.ndarray
    item_features: np.ndarray
    context_features: np.ndarray
    cross_features: np.ndarray
    aov_lift: float = 0.0
    is_chain: bool = True
    session_timestamp: Optional[pd.Timestamp] = None


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASETS
# ─────────────────────────────────────────────────────────────────────────────

class CSAODataset(Dataset):
    """
    PyTorch Dataset for Stage 2 DCTN ranking model.
    Wraps a list of TrainingExamples.
    """

    def __init__(self, examples: List[TrainingExample], max_cart_len: int = 20):
        self.examples    = examples
        self.max_cart_len = max_cart_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        # Pad / truncate cart sequence to fixed length
        cart_seq = ex.cart_item_ids[-self.max_cart_len:]
        pad_len  = self.max_cart_len - len(cart_seq)
        cart_padded = [0] * pad_len + cart_seq          # 0 = PAD token
        attention_mask = [0] * pad_len + [1] * len(cart_seq)

        return {
            "cart_ids":        torch.tensor(cart_padded, dtype=torch.long),
            "attention_mask":  torch.tensor(attention_mask, dtype=torch.long),
            "candidate_id":    torch.tensor(ex.candidate_item_id, dtype=torch.long),
            "user_features":   torch.tensor(ex.user_features, dtype=torch.float),
            "item_features":   torch.tensor(ex.item_features, dtype=torch.float),
            "context_features":torch.tensor(ex.context_features, dtype=torch.float),
            "cross_features":  torch.tensor(ex.cross_features, dtype=torch.float),
            "label":           torch.tensor(ex.label, dtype=torch.float),
            "aov_lift":        torch.tensor(ex.aov_lift, dtype=torch.float),
            "is_chain":        torch.tensor(int(ex.is_chain), dtype=torch.long),
        }


class TwoTowerDataset(Dataset):
    """
    PyTorch Dataset for Stage 1 Two-Tower contrastive training.
    Each example is (query_features, positive_item_id).
    In-batch negatives are handled inside the loss function.
    """

    def __init__(self, sessions_df: pd.DataFrame,
                 user_features: Dict[str, np.ndarray],
                 item_features: Dict[str, np.ndarray],
                 min_batch_size: int = 512):
        # Filter sessions with enough positive signals
        self.sessions     = sessions_df.reset_index(drop=True)
        self.user_features = user_features
        self.item_features = item_features
        self.min_batch_size = min_batch_size

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.sessions.iloc[idx]
        uf  = self.user_features.get(row["user_id"], np.zeros(64))
        # Use first accepted item as the positive key
        pos_item = row["items_ordered"][0] if row["items_ordered"] else "UNK"
        itf = self.item_features.get(pos_item, np.zeros(64))

        return {
            "query_features": torch.tensor(uf, dtype=torch.float),
            "pos_item_features": torch.tensor(itf, dtype=torch.float),
            "pos_item_id": pos_item,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL TRAIN / VAL / TEST SPLIT  (§5.1 — prevents data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(
    examples: List[TrainingExample],
    val_days: int = 7,
    test_days: int = 7,
) -> tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
    """
    Split training examples by timestamp:
      Train  → all before T − (val_days + test_days)
      Val    → T − (val_days + test_days)  to  T − test_days
      Test   → T − test_days  to  T

    This is the ONLY correct split for recommendation systems.
    Random splits cause future-data leakage into training.
    """
    timestamped = [ex for ex in examples if ex.session_timestamp is not None]
    timestamped.sort(key=lambda x: x.session_timestamp)

    if not timestamped:
        raise ValueError("No examples with session_timestamp found. Cannot do temporal split.")

    T = timestamped[-1].session_timestamp
    val_cutoff  = T - pd.Timedelta(days=val_days + test_days)
    test_cutoff = T - pd.Timedelta(days=test_days)

    train = [ex for ex in timestamped if ex.session_timestamp <  val_cutoff]
    val   = [ex for ex in timestamped if val_cutoff  <= ex.session_timestamp < test_cutoff]
    test  = [ex for ex in timestamped if ex.session_timestamp >= test_cutoff]

    print(f"[TemporalSplit] Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"               Cutoffs: val={val_cutoff.date()}, test={test_cutoff.date()}, T={T.date()}")
    return train, val, test
