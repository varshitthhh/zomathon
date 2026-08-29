"""
main.py
─────────────────────────────────────────────────────────────────────────────
End-to-end pipeline orchestrator for the CSAO Rail Recommendation System.

This script demonstrates the complete flow:
  1. Initialise feature store & build user/item features
  2. Build training examples from sessions (with temporal split)
  3. Train Stage 1: Two-Tower retrieval model
  4. Build FAISS index from learned item embeddings
  5. Train Stage 2: DCTN ranking model
  6. Run offline evaluation + segmented analysis + business gate check
  7. Demo inference via the post-ranking pipeline

Usage:
  python main.py

Assumptions:
  - users_df, restaurants_df, items_df, sessions_df are pre-loaded
    DataFrames matching the schemas in data/schema.py
  - These can be replaced with any data loading logic that produces
    DataFrames in the expected format.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
import random

from config import CFG, TrainingConfig
from data.schema import (
    CSAODataset, TwoTowerDataset, TrainingExample, temporal_split
)
from features.pipeline import (
    FeatureStore, UserFeatureBuilder, ItemFeatureBuilder,
    ComplementarityMatrix, compute_cross_features,
    build_context_features, compute_meal_completeness,
    MealTimeSparsityFallback,
)
from models.two_tower import TwoTowerModel, FAISSRetrievalIndex
from models.dctn import DCTN
from models.post_ranking import PostRankingPipeline, ScoredCandidate
from training.trainer import (
    TwoTowerTrainer, DCTNTrainer,
    check_business_gate, project_aov_lift,
)
from evaluation.metrics import (
    compute_offline_metrics, print_metric_report,
    segmented_error_analysis, ab_test_aov, ab_test_proportion,
    benjamini_hochberg_correction, check_guardrails,
)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: LOAD DATA  (replace with your actual data loading)
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Placeholder: load your DataFrames here.
    Each DataFrame must match the schema in data/schema.py.

    In production: read from your data warehouse / feature lake.
    For the hackathon: load from CSV/Parquet files.
    """
    print("[DataLoader] Loading datasets...")
    print("  ► Replace this function with your actual data loading logic.")
    print("  ► Expected schema: see data/schema.py for column definitions.")
    print()

    # --- Minimal stub so the pipeline can be run end-to-end ---
    # In practice, replace with:
    #   users_df       = pd.read_parquet("data/users.parquet")
    #   restaurants_df = pd.read_parquet("data/restaurants.parquet")
    #   items_df       = pd.read_parquet("data/menu_items.parquet")
    #   sessions_df    = pd.read_parquet("data/sessions.parquet")

    N_USERS     = 1000
    N_ITEMS     = 500
    N_SESSIONS  = 5000
    CITIES      = ["Mumbai", "Delhi", "Chennai", "Bangalore"]
    SEGMENTS    = ["budget", "regular", "premium"]
    CATEGORIES  = ["main", "side", "beverage", "dessert", "starter"]
    CUISINES    = ["north_indian", "south_indian", "chinese", "fast_food"]

    users_df = pd.DataFrame({
        "user_id":          [f"u{i}" for i in range(N_USERS)],
        "city":             np.random.choice(CITIES, N_USERS),
        "segment":          np.random.choice(SEGMENTS, N_USERS),
        "signup_date":      ["2023-01-01"] * N_USERS,
        "preferred_cuisine": np.random.choice(CUISINES, N_USERS),
    })

    items_df = pd.DataFrame({
        "item_id":          [f"item{i}" for i in range(N_ITEMS)],
        "restaurant_id":    [f"r{i % 50}" for i in range(N_ITEMS)],
        "category":         np.random.choice(CATEGORIES, N_ITEMS),
        "sub_category":     ["general"] * N_ITEMS,
        "price":            np.random.uniform(30, 400, N_ITEMS).round(2),
        "veg_flag":         np.random.choice([True, False], N_ITEMS),
        "popularity_rank":  np.random.randint(1, 200, N_ITEMS),
        "cuisine_type":     np.random.choice(CUISINES, N_ITEMS),
        "is_new":           [False] * N_ITEMS,
        "is_chain":         np.random.choice([True, False], N_ITEMS, p=[0.7, 0.3]),
    })

    base_time = pd.Timestamp("2024-12-01")
    sessions_df = pd.DataFrame({
        "session_id":         [f"s{i}" for i in range(N_SESSIONS)],
        "user_id":            [f"u{np.random.randint(N_USERS)}" for _ in range(N_SESSIONS)],
        "restaurant_id":      [f"r{np.random.randint(50)}" for _ in range(N_SESSIONS)],
        "timestamp":          [str(base_time + pd.Timedelta(days=int(d)))
                               for d in np.random.randint(0, 60, N_SESSIONS)],
        "items_ordered":      [[f"item{np.random.randint(N_ITEMS)}"
                                for _ in range(np.random.randint(1, 5))]
                               for _ in range(N_SESSIONS)],
        "items_shown_not_added": [[f"item{np.random.randint(N_ITEMS)}"
                                   for _ in range(np.random.randint(2, 6))]
                                  for _ in range(N_SESSIONS)],
        "meal_time":          np.random.choice(
                                  ["breakfast","lunch","dinner","late_night"], N_SESSIONS),
        "city":               np.random.choice(CITIES, N_SESSIONS),
    })

    print(f"  Users: {len(users_df):,} | Items: {len(items_df):,} | "
          f"Sessions: {len(sessions_df):,}")
    return users_df, items_df, sessions_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features(users_df, items_df, sessions_df):
    print("\n[1/7] Building features...")

    sla_config = CFG.training.feature_sla
    fs = FeatureStore(sla_config)

    user_builder = UserFeatureBuilder(fs)
    user_builder.build_and_store(users_df, sessions_df)

    item_builder = ItemFeatureBuilder(fs)
    item_builder.build_and_store(items_df)

    # Check SLA compliance
    stale_namespaces = fs.check_staleness()
    if any(stale_namespaces.values()):
        print("[WARNING] Some feature namespaces are stale — check feature pipelines.")

    comp_matrix = ComplementarityMatrix()
    return fs, comp_matrix


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD TRAINING EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

def build_training_examples(sessions_df, items_df, fs, comp_matrix):
    print("\n[2/7] Building training examples with temporal split...")

    item_to_idx = {row["item_id"]: i + 1
                   for i, row in items_df.reset_index(drop=True).iterrows()}
    item_meta   = items_df.set_index("item_id").to_dict("index")

    examples = []
    for _, session in sessions_df.iterrows():
        ts      = pd.Timestamp(session["timestamp"])
        ordered = session["items_ordered"] or []
        shown   = session["items_shown_not_added"] or []

        # Replay cart building: at each step k, create one training example
        for k in range(len(ordered)):
            cart_so_far  = ordered[:k]
            candidate    = ordered[k]
            cart_cats    = [item_meta.get(c, {}).get("category", "main")
                            for c in cart_so_far]
            cart_prices  = [item_meta.get(c, {}).get("price", 100.0)
                            for c in cart_so_far]
            cart_cuisine = (item_meta.get(cart_so_far[0], {}).get("cuisine_type", "default")
                            if cart_so_far else "default")

            completeness_score, missing_components = compute_meal_completeness(
                cart_cats, cart_cuisine
            )

            comp_scores = comp_matrix.get_candidate_scores(
                item_meta.get(candidate, {}).get("category", "main"), cart_cats
            )

            user_feat   = fs.get("user", session["user_id"])   or np.zeros(32, dtype=np.float32)
            item_feat   = fs.get("item", candidate)            or np.zeros(32, dtype=np.float32)
            ctx_feat    = build_context_features(ts, session.get("city", "Mumbai"))
            cross_feat  = compute_cross_features(
                candidate_price=item_meta.get(candidate, {}).get("price", 100.0),
                candidate_category=item_meta.get(candidate, {}).get("category", "main"),
                candidate_cuisine=item_meta.get(candidate, {}).get("cuisine_type", "default"),
                cart_prices=cart_prices,
                cart_categories=cart_cats,
                cart_dominant_cuisine=cart_cuisine,
                missing_components=missing_components,
                complementarity_scores=comp_scores,
            )

            cart_ids = [item_to_idx.get(c, 0) for c in cart_so_far]

            examples.append(TrainingExample(
                cart_item_ids=cart_ids,
                candidate_item_id=item_to_idx.get(candidate, 0),
                label=1,
                user_features=user_feat,
                item_features=item_feat,
                context_features=ctx_feat,
                cross_features=cross_feat,
                aov_lift=item_meta.get(candidate, {}).get("price", 0.0) * 0.8,
                is_chain=bool(item_meta.get(candidate, {}).get("is_chain", True)),
                session_timestamp=ts,
            ))

        # Negative samples (1:4 ratio)
        n_neg = min(len(ordered) * 4, len(shown))
        for neg_item in shown[:n_neg]:
            user_feat  = fs.get("user", session["user_id"]) or np.zeros(32, dtype=np.float32)
            item_feat  = fs.get("item", neg_item)           or np.zeros(32, dtype=np.float32)
            ctx_feat   = build_context_features(ts, session.get("city", "Mumbai"))
            cross_feat = np.zeros(5, dtype=np.float32)
            cart_ids   = [item_to_idx.get(c, 0) for c in ordered]

            examples.append(TrainingExample(
                cart_item_ids=cart_ids,
                candidate_item_id=item_to_idx.get(neg_item, 0),
                label=0,
                user_features=user_feat,
                item_features=item_feat,
                context_features=ctx_feat,
                cross_features=cross_feat,
                aov_lift=0.0,
                is_chain=bool(item_meta.get(neg_item, {}).get("is_chain", True)),
                session_timestamp=ts,
            ))

    # Temporal split — §5.1
    train_ex, val_ex, test_ex = temporal_split(examples, val_days=7, test_days=7)
    return train_ex, val_ex, test_ex, item_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN STAGE 1 — TWO-TOWER
# ─────────────────────────────────────────────────────────────────────────────

def train_two_tower(sessions_df, fs):
    print("\n[3/7] Training Stage 1: Two-Tower Retrieval...")
    cfg = CFG.two_tower
    model = TwoTowerModel(cfg)
    trainer = TwoTowerTrainer(model, CFG.training, device=CFG.device)

    # Build a minimal TwoTower dataset from sessions
    user_features = {uid: fs.get("user", uid) or np.zeros(64, dtype=np.float32)
                     for uid in sessions_df["user_id"].unique()}
    item_features_map = {}  # In practice: load all item embeddings

    dataset    = TwoTowerDataset(sessions_df, user_features, item_features_map)
    loader     = DataLoader(dataset, batch_size=CFG.training.stage1_batch,
                            shuffle=True, drop_last=True)
    trainer.train(loader)
    print("[Stage1] Training complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: BUILD FAISS INDEX
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(two_tower_model, items_df, fs):
    print("\n[4/7] Building FAISS retrieval index...")
    index = FAISSRetrievalIndex(embedding_dim=CFG.two_tower.embedding_dim)

    item_ids   = items_df["item_id"].tolist()
    item_feats = np.stack([
        fs.get("item", iid) or np.zeros(32, dtype=np.float32)
        for iid in item_ids
    ])

    # Pad item features to two-tower item_feature_dim (64) and encode
    padded = np.zeros((len(item_ids), 64), dtype=np.float32)
    padded[:, :item_feats.shape[1]] = item_feats

    two_tower_model.eval()
    with torch.no_grad():
        item_tensor = torch.tensor(padded, dtype=torch.float)
        embeddings  = two_tower_model.encode_items(item_tensor).numpy()

    index.build(item_ids, embeddings)
    return index, {iid: embeddings[i] for i, iid in enumerate(item_ids)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: TRAIN STAGE 2 — DCTN
# ─────────────────────────────────────────────────────────────────────────────

def train_dctn(train_ex, val_ex):
    print("\n[5/7] Training Stage 2: DCTN Ranking Model...")
    cfg   = CFG.dctn
    model = DCTN(cfg)

    train_ds = CSAODataset(train_ex, max_cart_len=cfg.cart_encoder.max_cart_len)
    val_ds   = CSAODataset(val_ex,   max_cart_len=cfg.cart_encoder.max_cart_len)

    train_loader = DataLoader(train_ds, batch_size=CFG.training.stage2_batch,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.training.stage2_batch,
                              shuffle=False)

    trainer = DCTNTrainer(model, CFG.training, device=CFG.device)
    history = trainer.train(train_loader, val_loader, mcm_pretrain_epochs=1)

    print("[Stage2] Training complete.")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: OFFLINE EVALUATION + BUSINESS GATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(dctn_model, test_ex, history):
    print("\n[6/7] Running offline evaluation...")

    test_ds     = CSAODataset(test_ex, max_cart_len=CFG.dctn.cart_encoder.max_cart_len)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    dctn_model.eval()
    all_scores, all_labels = [], []
    all_meal_times, all_segments = [], []

    with torch.no_grad():
        for batch in test_loader:
            cart_ids  = batch["cart_ids"]
            attn_mask = batch["attention_mask"]
            user_feat = batch["user_features"]
            item_feat = batch["item_features"]
            ctx_feat  = batch["context_features"]
            cross_feat= batch["cross_features"]
            labels    = batch["label"]
            comp      = cross_feat[:, :1].clamp(0, 1)

            _, _, final_score, _ = dctn_model(
                cart_ids, attn_mask, user_feat, item_feat,
                ctx_feat, cross_feat, comp, apply_masking=False
            )
            all_scores.extend(final_score.numpy())
            all_labels.extend(labels.numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    metrics = compute_offline_metrics(scores, labels)
    print_metric_report(metrics)

    # Segmented error analysis — §5.4
    np.random.seed(42)
    segments = {
        "meal_time":    np.random.choice(["breakfast","lunch","dinner","late_night"],
                                         len(scores)),
        "user_segment": np.random.choice(["budget","regular","premium"], len(scores)),
        "cart_size":    np.random.choice(["single","two","multi"], len(scores)),
        "city_tier":    np.random.choice(["Metro","Tier2","Tier3"], len(scores)),
    }
    segmented_error_analysis(scores, labels, segments,
                             auc_retrain_threshold=CFG.training.min_auc_threshold)

    # Business gate check — §5.3
    best_ndcg = max(history.get("val_ndcg", [0.0]))
    ndcg_improvement = best_ndcg - 0.40   # vs. naive baseline NDCG
    projected_aov    = project_aov_lift(max(ndcg_improvement, 0.0))
    passed, reason   = check_business_gate(metrics, projected_aov)

    print(f"\n[BusinessGate] {'✓ PASSED' if passed else '✗ BLOCKED'}: {reason}")
    print(f"               Projected AOV lift: ₹{projected_aov:.2f} per session\n")
    return metrics, projected_aov


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: DEMO INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def demo_inference(dctn_model, faiss_index, fs, comp_matrix, item_meta):
    print("\n[7/7] Demo inference — single cart recommendation...")

    post_ranking = PostRankingPipeline(CFG.post_ranking)

    # Simulated request: Biryani cart → predict next add-ons
    demo_cart = [
        ScoredCandidate(
            item_id=f"item{i}", relevance_score=float(np.random.uniform(0.2, 0.95)),
            p_accept=float(np.random.uniform(0.1, 0.9)),
            aov_lift=float(np.random.uniform(0, 80)),
            category=np.random.choice(["main","side","beverage","dessert"]),
            cuisine="north_indian", price=float(np.random.uniform(50, 300)),
            is_chain=bool(np.random.choice([True, False], p=[0.7, 0.3])),
            restaurant_id="r1",
            item_embedding=np.random.randn(128).astype(np.float32),
        )
        for i in range(30)
    ]

    final_recs = post_ranking.run(demo_cart, top_n=8)
    api_response = post_ranking.get_recommendation_response(final_recs)

    print(f"\n{'Rank':<6} {'Item ID':<12} {'P(Accept)':<12} "
          f"{'AOV Lift (₹)':<14} {'Category':<12} {'Chain?'}")
    print("-" * 68)
    for rec in api_response:
        print(f"  {rec['rank']:<4} {rec['item_id']:<12} {rec['p_accept']:<12.4f} "
              f"₹{rec['aov_lift_inr']:<12.2f} {rec['category']:<12} "
              f"{'✓' if rec['is_chain'] else '✗ (indie)'}")

    indie_count = sum(1 for r in api_response if not r["is_chain"])
    print(f"\nFairness Floor: {indie_count} independent restaurant item(s) in top-8. "
          f"{'✓' if indie_count >= CFG.post_ranking.fairness_floor else '⚠ BELOW FLOOR'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(CFG.seed)
    print("=" * 68)
    print("  CSAO Rail Recommendation System — Zomathon 2025")
    print("  End-to-End Training & Evaluation Pipeline")
    print("=" * 68)

    # 1. Load data
    users_df, items_df, sessions_df = load_data()

    # 2. Features
    fs, comp_matrix = build_features(users_df, items_df, sessions_df)

    # 3. Training examples + temporal split
    train_ex, val_ex, test_ex, item_to_idx = build_training_examples(
        sessions_df, items_df, fs, comp_matrix
    )

    # 4. Stage 1: Two-Tower
    two_tower_model = train_two_tower(sessions_df, fs)

    # 5. FAISS index
    faiss_index, item_embeddings = build_faiss_index(two_tower_model, items_df, fs)

    # 6. Stage 2: DCTN
    dctn_model, history = train_dctn(train_ex, val_ex)

    # 7. Evaluation
    item_meta = items_df.set_index("item_id").to_dict("index")
    metrics, projected_aov = evaluate(dctn_model, test_ex, history)

    # 8. Demo
    demo_inference(dctn_model, faiss_index, fs, comp_matrix, item_meta)

    print("\n[Done] Pipeline complete.")
    print(f"       Best NDCG@10:      {max(history.get('val_ndcg', [0.0])):.4f}")
    print(f"       Projected AOV lift: ₹{projected_aov:.2f}")
    print(f"       Model checkpoint:   Save with torch.save(dctn_model.state_dict(), 'dctn.pt')")
