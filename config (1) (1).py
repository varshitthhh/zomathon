"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the CSAO Rail Recommendation System.
All hyperparameters, thresholds, and constants are defined here.
"""

from dataclasses import dataclass, field
from typing import List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# DATA SCHEMA CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MEAL_BLUEPRINTS: Dict[str, List[str]] = {
    "north_indian": ["main", "bread_rice", "side_gravy", "beverage", "dessert"],
    "south_indian": ["main", "accompaniment", "beverage", "dessert"],
    "chinese":      ["main", "side", "soup", "beverage"],
    "mughlai":      ["main", "bread", "side_gravy", "beverage", "dessert"],
    "fast_food":    ["main", "side", "beverage"],
    "pizza":        ["main", "side", "beverage", "dip"],
    "default":      ["main", "side", "beverage"],
}

MEAL_TIME_WINDOWS = {
    "breakfast":  (7,  10),
    "lunch":      (12, 15),
    "snack":      (15, 18),
    "dinner":     (19, 22),
    "late_night": (22, 3),
}

CITY_CUISINE_AFFINITY = {
    "Chennai":   {"south_indian": 0.9, "north_indian": 0.3, "chinese": 0.5},
    "Delhi":     {"north_indian": 0.9, "mughlai": 0.8, "chinese": 0.5},
    "Mumbai":    {"fast_food": 0.7, "north_indian": 0.6, "chinese": 0.6},
    "Bangalore": {"south_indian": 0.7, "north_indian": 0.5, "chinese": 0.6},
    "Lucknow":   {"mughlai": 0.9, "north_indian": 0.7},
}

USER_SEGMENTS = ["budget", "regular", "premium"]
CITY_TIERS    = {"Metro": ["Mumbai", "Delhi", "Bangalore", "Chennai"],
                 "Tier2": ["Lucknow", "Pune", "Ahmedabad"],
                 "Tier3": []}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TwoTowerConfig:
    """Stage 1: Two-Tower Retrieval Model."""
    user_feature_dim: int   = 64
    item_feature_dim: int   = 64
    embedding_dim: int      = 128       # Output tower dimension
    hidden_dims: List[int]  = field(default_factory=lambda: [256, 128])
    temperature: float      = 0.07      # InfoNCE learnable temperature init
    min_batch_size: int     = 512
    top_k_retrieval: int    = 100       # Candidates passed to Stage 2
    faiss_nprobe: int       = 32        # FAISS IVF probe count
    dropout: float          = 0.1


@dataclass
class CartEncoderConfig:
    """Masked Cart Encoder (Bidirectional Transformer)."""
    item_vocab_size: int    = 50_001    # +1 for PAD token
    item_embed_dim: int     = 64
    num_heads: int          = 4
    num_layers: int         = 2
    ffn_dim: int            = 256
    max_cart_len: int       = 20        # Max items in a cart
    mask_prob: float        = 0.15      # MCM masking probability
    dropout: float          = 0.1
    pad_token_id: int       = 0
    mask_token_id: int      = 50_000


@dataclass
class DCNv2Config:
    """Deep & Cross Network v2."""
    input_dim: int          = 256       # Matches concatenated feature dim
    cross_layers: int       = 3
    deep_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float          = 0.1


@dataclass
class MMoEConfig:
    """Multi-Gate Mixture of Experts scoring head."""
    input_dim: int          = 128       # From DCN-v2 output
    num_experts: int        = 4
    expert_hidden_dim: int  = 64
    num_tasks: int          = 2         # Task1: P(Accept), Task2: AOV Lift
    alpha: float            = 0.6       # Score = alpha*P(Accept) + (1-alpha)*AOV
    dropout: float          = 0.1


@dataclass
class DCTNConfig:
    """Full DCTN Ranking Model config."""
    cart_encoder: CartEncoderConfig = field(default_factory=CartEncoderConfig)
    dcn: DCNv2Config                = field(default_factory=DCNv2Config)
    mmoe: MMoEConfig                = field(default_factory=MMoEConfig)
    user_feature_dim: int           = 32
    item_feature_dim: int           = 32
    context_feature_dim: int        = 16
    cross_feature_dim: int          = 16


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Data split
    val_days: int           = 7
    test_days: int          = 7

    # Stage 1
    stage1_epochs: int      = 10
    stage1_lr: float        = 1e-3
    stage1_batch: int       = 512
    stage1_weight_decay: float = 1e-4

    # Stage 2
    stage2_epochs: int      = 20
    stage2_lr: float        = 5e-4
    stage2_batch: int       = 256
    stage2_weight_decay: float = 1e-4
    lambda_aov: float       = 0.3       # AOV loss weight in composite loss
    dropout: float          = 0.2
    early_stopping_patience: int = 3

    # Thresholds
    min_auc_threshold: float        = 0.70   # Re-train slice if below this
    min_projected_aov_lift: float   = 5.0    # Gate for shadow mode (₹)

    # Feature refresh SLAs (hours)
    feature_sla: Dict[str, float] = field(default_factory=lambda: {
        "user":       26.0,
        "restaurant": 1.5,
        "item":       26.0,
        "real_time":  0.0,
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST-RANKING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PostRankingConfig:
    top_n: int              = 8         # Items shown on CSAO rail
    mmr_lambda: float       = 0.7       # λ in MMR: relevance vs diversity
    fairness_floor: int     = 1         # Min independent restaurant items in top-N
    fairness_position: int  = 7         # Insert at this position if floor not met


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM / SERVING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ServingConfig:
    latency_budget_ms: int          = 200
    circuit_breaker_ms: int         = 250
    redis_ttl_seconds: int          = 7200      # 2 hours
    cache_hit_target: float         = 0.99
    request_coalesce_window_ms: int = 500
    stale_while_revalidate: bool    = True


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    two_tower: TwoTowerConfig       = field(default_factory=TwoTowerConfig)
    dctn: DCTNConfig                = field(default_factory=DCTNConfig)
    training: TrainingConfig        = field(default_factory=TrainingConfig)
    post_ranking: PostRankingConfig = field(default_factory=PostRankingConfig)
    serving: ServingConfig          = field(default_factory=ServingConfig)
    device: str                     = "cpu"     # "cuda" if GPU available
    seed: int                       = 42


CFG = Config()
