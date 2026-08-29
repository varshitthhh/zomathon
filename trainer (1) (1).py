"""
training/trainer.py
─────────────────────────────────────────────────────────────────────────────
Training pipeline for Stage 1 (Two-Tower) and Stage 2 (DCTN) — §5.1, §5.2

Implements:
  - Temporal data split (prevents data leakage — §5.1)
  - Stage 1: InfoNCE contrastive training
  - Stage 2: Composite loss (LambdaRank + AOV MSE + MCM)
  - Early stopping on validation NDCG@10
  - Business gate: model not promoted if projected AOV lift < ₹5 (§5.3)
  - Segmented error analysis (§5.4)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import copy

from models.two_tower import TwoTowerModel, infonce_loss
from models.dctn import DCTN, CompositeLoss
from evaluation.metrics import compute_offline_metrics, segmented_error_analysis
from config import TrainingConfig, TwoTowerConfig, DCTNConfig, CFG


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 TRAINER  — Two-Tower Contrastive
# ─────────────────────────────────────────────────────────────────────────────

class TwoTowerTrainer:
    """
    Trains the Two-Tower retrieval model with InfoNCE loss.

    Batch construction ensures sessions are shuffled across restaurants
    and cities — making in-batch negatives genuinely hard (§4.2).
    """

    def __init__(
        self,
        model: TwoTowerModel,
        cfg: TrainingConfig,
        device: str = "cpu",
    ):
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device
        self.optim  = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.stage1_lr,
            weight_decay=cfg.stage1_weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=cfg.stage1_epochs
        )

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> dict:
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_count = 0

        for epoch in range(self.cfg.stage1_epochs):
            # ── Training ──
            self.model.train()
            epoch_losses = []

            for batch in tqdm(train_loader, desc=f"[Stage1 Epoch {epoch+1}]", leave=False):
                # Drop batches below minimum size (§4.2)
                if batch["query_features"].shape[0] < self.cfg.stage1_batch:
                    continue

                q = batch["query_features"].to(self.device)
                k = batch["pos_item_features"].to(self.device)

                self.optim.zero_grad()
                loss = self.model(q, k)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                epoch_losses.append(loss.item())

            train_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            history["train_loss"].append(train_loss)

            # ── Validation ──
            val_loss = float("nan")
            if val_loader:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.cfg.early_stopping_patience:
                        print(f"[Stage1] Early stopping at epoch {epoch+1}.")
                        break

            self.scheduler.step()
            print(f"[Stage1] Epoch {epoch+1:02d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        return history

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                if batch["query_features"].shape[0] < 2:
                    continue
                q = batch["query_features"].to(self.device)
                k = batch["pos_item_features"].to(self.device)
                loss = self.model(q, k)
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 TRAINER  — DCTN + Composite Loss
# ─────────────────────────────────────────────────────────────────────────────

class DCTNTrainer:
    """
    Trains the DCTN ranking model with the composite loss (§5.2).

    Training phases:
      Phase 1 (optional MCM pre-training): Train cart encoder with MCM only.
      Phase 2 (joint training): Train full model with composite loss.
    """

    def __init__(
        self,
        model: DCTN,
        cfg: TrainingConfig,
        device: str = "cpu",
    ):
        self.model    = model.to(device)
        self.cfg      = cfg
        self.device   = device
        self.criterion = CompositeLoss(
            lambda_aov=cfg.lambda_aov,
            lambda_mcm=0.1,
        )
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.stage2_lr,
            weight_decay=cfg.stage2_weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode="max", patience=2, factor=0.5
        )
        self.best_model_state = None
        self.best_ndcg        = -float("inf")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mcm_pretrain_epochs: int = 2,
    ) -> dict:
        history = {"train_loss": [], "val_ndcg": [], "val_auc": []}

        # ── Phase 1: MCM Pre-training ──────────────────────────────────────
        if mcm_pretrain_epochs > 0:
            print("[Stage2] MCM pre-training phase...")
            self._mcm_pretrain(train_loader, mcm_pretrain_epochs)

        # ── Phase 2: Joint Training ─────────────────────────────────────────
        patience_count = 0

        for epoch in range(self.cfg.stage2_epochs):
            train_loss_info = self._train_epoch(train_loader, apply_masking=False)
            val_metrics     = self._validate_epoch(val_loader)

            history["train_loss"].append(train_loss_info["loss_total"])
            history["val_ndcg"].append(val_metrics.get("ndcg_at_10", 0.0))
            history["val_auc"].append(val_metrics.get("auc", 0.0))

            val_ndcg = val_metrics.get("ndcg_at_10", 0.0)
            self.scheduler.step(val_ndcg)

            print(f"[Stage2] Epoch {epoch+1:02d} | "
                  f"Loss: {train_loss_info['loss_total']:.4f} | "
                  f"NDCG@10: {val_ndcg:.4f} | "
                  f"AUC: {val_metrics.get('auc', 0.0):.4f} | "
                  f"P@8: {val_metrics.get('precision_at_8', 0.0):.4f}")

            if val_ndcg > self.best_ndcg:
                self.best_ndcg = val_ndcg
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.cfg.early_stopping_patience:
                    print(f"[Stage2] Early stopping at epoch {epoch+1}.")
                    break

        # Restore best checkpoint
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"[Stage2] Restored best model (NDCG@10={self.best_ndcg:.4f}).")

        return history

    def _mcm_pretrain(self, loader: DataLoader, epochs: int):
        """Pre-train cart encoder via Masked Cart Modeling."""
        # Only optimise the cart encoder parameters
        encoder_params = list(self.model.cart_encoder.parameters())
        pretrain_optim = torch.optim.AdamW(encoder_params, lr=1e-3)

        for epoch in range(epochs):
            self.model.train()
            losses = []
            for batch in tqdm(loader, desc=f"[MCM Pretrain {epoch+1}]", leave=False):
                cart_ids = batch["cart_ids"].to(self.device)
                attn_msk = batch["attention_mask"].to(self.device)

                # Run encoder with masking
                _, mlm_logits = self.model.cart_encoder(
                    cart_ids, attn_msk, apply_masking=True
                )

                # MCM loss: reconstruct masked items
                if mlm_logits is not None:
                    mcm_mask   = (cart_ids == self.model.cart_encoder.cfg.mask_token_id)
                    masked_log = mlm_logits[mcm_mask]
                    masked_tgt = cart_ids[mcm_mask]   # Original IDs before masking
                    if masked_log.shape[0] > 0:
                        loss = torch.nn.functional.cross_entropy(masked_log, masked_tgt)
                        pretrain_optim.zero_grad()
                        loss.backward()
                        pretrain_optim.step()
                        losses.append(loss.item())

            print(f"[MCM] Epoch {epoch+1} | Loss: {np.mean(losses):.4f}")

    def _train_epoch(self, loader: DataLoader, apply_masking: bool = False) -> dict:
        self.model.train()
        total_loss_info = {k: 0.0 for k in
                           ["loss_total", "loss_rank", "loss_aov", "loss_bce", "loss_mcm"]}
        n_batches = 0

        for batch in tqdm(loader, desc="[Training]", leave=False):
            self.optim.zero_grad()

            outputs = self._forward_batch(batch, apply_masking)
            loss, breakdown = self.criterion(**outputs)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optim.step()

            for k in total_loss_info:
                total_loss_info[k] += breakdown.get(k, 0.0)
            n_batches += 1

        if n_batches > 0:
            for k in total_loss_info:
                total_loss_info[k] /= n_batches

        return total_loss_info

    def _validate_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_scores, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                outputs = self._forward_batch(batch, apply_masking=False)
                all_scores.extend(outputs["final_score"].cpu().numpy())
                all_labels.extend(outputs["labels"].cpu().numpy())

        if not all_scores:
            return {}

        scores = np.array(all_scores)
        labels = np.array(all_labels)
        return compute_offline_metrics(scores, labels)

    def _forward_batch(self, batch: dict, apply_masking: bool) -> dict:
        """Run a batch through the DCTN model."""
        cart_ids       = batch["cart_ids"].to(self.device)
        attn_mask      = batch["attention_mask"].to(self.device)
        user_feat      = batch["user_features"].to(self.device)
        item_feat      = batch["item_features"].to(self.device)
        ctx_feat       = batch["context_features"].to(self.device)
        cross_feat     = batch["cross_features"].to(self.device)
        labels         = batch["label"].to(self.device)
        aov_lift_true  = batch["aov_lift"].to(self.device)

        # Compute meal completeness from cross_features[:, 0] (category_fills_missing)
        # In production this is a pre-computed feature; here we approximate
        completeness   = cross_feat[:, :1].clamp(0.0, 1.0)   # (B, 1)

        p_accept, aov_lift, final_score, mlm_logits = self.model(
            cart_ids=cart_ids,
            attention_mask=attn_mask,
            user_features=user_feat,
            item_features=item_feat,
            context_features=ctx_feat,
            cross_features=cross_feat,
            completeness_score=completeness,
            apply_masking=apply_masking,
        )

        return {
            "final_score":    final_score,
            "p_accept":       p_accept,
            "aov_lift_pred":  aov_lift,
            "labels":         labels,
            "aov_lift_true":  aov_lift_true,
            "mlm_logits":     mlm_logits,
        }


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS METRIC GATE  (§5.3 — New Addition)
# ─────────────────────────────────────────────────────────────────────────────

def check_business_gate(
    val_metrics: dict,
    projected_aov_lift: float,
    min_auc: float = 0.70,
    min_aov_lift: float = 5.0,
) -> Tuple[bool, str]:
    """
    §5.3 — Gate for promotion to shadow mode.

    Rules:
      1. AUC must be >= min_auc
      2. Projected AOV lift must be >= min_aov_lift (₹)

    Returns: (passed: bool, reason: str)
    """
    auc = val_metrics.get("auc", 0.0)
    if auc < min_auc:
        return False, f"AUC {auc:.3f} < threshold {min_auc} — model not promoted."

    if projected_aov_lift < min_aov_lift:
        return False, (f"Projected AOV lift ₹{projected_aov_lift:.1f} < "
                       f"₹{min_aov_lift} gate — model not promoted to shadow mode.")

    return True, (f"Business gate PASSED: AUC={auc:.3f}, "
                  f"AOV lift=₹{projected_aov_lift:.1f}")


def project_aov_lift(
    ndcg_improvement: float,
    base_acceptance_rate: float = 0.20,
    avg_addon_price: float = 80.0,
    monthly_sessions: int = 2_000_000,
) -> float:
    """
    Project AOV lift from offline NDCG improvement (§7.5).

    Rule of thumb: 0.05 NDCG improvement → ~2% acceptance rate improvement.
    At 2M monthly sessions, 20% base rate:
      Additional add-ons = sessions × Δacceptance_rate
      Monthly GMV lift   = additional_add_ons × avg_addon_price
      Per-session AOV    = Monthly GMV lift / sessions
    """
    acceptance_lift = (ndcg_improvement / 0.05) * 0.02
    additional_addons = monthly_sessions * acceptance_lift
    monthly_gmv_lift  = additional_addons * avg_addon_price
    per_session_aov   = monthly_gmv_lift / monthly_sessions
    return round(per_session_aov, 2)
