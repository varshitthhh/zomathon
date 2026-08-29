"""
models/dctn.py
─────────────────────────────────────────────────────────────────────────────
Stage 2: Deep Cross Transformer Network (DCTN) — §4.3

Three components fused into one ranking model:
  A. Masked Cart Encoder  — Bidirectional Transformer (§4.3.1)
  B. Cross Feature Network — DCN-v2                   (§4.3.2)
  C. Multi-Gate Mixture of Experts (MMoE)             (§4.3.3)

Training objective: composite loss
  L_total = L_rank (LambdaRank/ListNet) + λ × L_aov (MSE)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from config import CartEncoderConfig, DCNv2Config, MMoEConfig, DCTNConfig


# ─────────────────────────────────────────────────────────────────────────────
# A. MASKED CART ENCODER  (§4.3.1)
# ─────────────────────────────────────────────────────────────────────────────

class MaskedCartEncoder(nn.Module):
    """
    Bidirectional Transformer encoder for cart sequences.

    Training: Masked Cart Modeling (MCM) — 15% of items randomly masked,
              model reconstructs masked item IDs (like BERT's MLM).
    Inference: Full sequence encoded bidirectionally. CLS token represents
               full cart intent including structurally missing components.

    Item removal handling: Encoder is stateless — re-run with updated
    sequence after any cart mutation (add or remove) in < 15ms.
    """

    def __init__(self, cfg: CartEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Item embedding table (+2 for PAD and MASK tokens)
        self.item_embedding = nn.Embedding(
            cfg.item_vocab_size + 1, cfg.item_embed_dim, padding_idx=cfg.pad_token_id
        )
        # CLS token embedding (prepended to every sequence)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, cfg.item_embed_dim))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(cfg.item_embed_dim, cfg.max_cart_len + 1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.item_embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # MLM head for MCM training objective
        self.mlm_head = nn.Linear(cfg.item_embed_dim, cfg.item_vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.item_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.cls_embedding, std=0.02)

    def apply_mcm_masking(
        self,
        item_ids: torch.Tensor,         # (B, L)
        mask_prob: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Masked Cart Modeling masking during training.
        Returns: (masked_ids, bool_mask) where bool_mask=True means masked.
        """
        masked_ids = item_ids.clone()
        bool_mask  = torch.zeros_like(item_ids, dtype=torch.bool)

        # Only mask non-PAD tokens
        eligible = (item_ids != self.cfg.pad_token_id)
        rand     = torch.rand_like(item_ids, dtype=torch.float)
        to_mask  = eligible & (rand < mask_prob)

        masked_ids[to_mask] = self.cfg.mask_token_id
        bool_mask[to_mask]  = True
        return masked_ids, bool_mask

    def forward(
        self,
        item_ids: torch.Tensor,         # (B, L)  — already padded
        attention_mask: torch.Tensor,   # (B, L)  — 1=real, 0=pad
        apply_masking: bool = False,    # True only during MCM training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            cls_output:   (B, item_embed_dim) — cart-level representation
            mlm_logits:   (B, L, vocab_size)  — only when apply_masking=True
        """
        B, L = item_ids.shape
        masked_ids = item_ids
        mcm_mask   = None

        if apply_masking:
            masked_ids, mcm_mask = self.apply_mcm_masking(item_ids)

        # Item embeddings + positional encoding
        embeds = self.item_embedding(masked_ids)       # (B, L, D)
        embeds = self.pos_encoding(embeds)

        # Prepend CLS token
        cls    = self.cls_embedding.expand(B, -1, -1)  # (B, 1, D)
        seq    = torch.cat([cls, embeds], dim=1)        # (B, L+1, D)

        # Build key padding mask (True = ignore position)
        cls_mask      = torch.zeros(B, 1, dtype=torch.bool, device=item_ids.device)
        pad_mask      = (attention_mask == 0)           # (B, L)
        src_key_mask  = torch.cat([cls_mask, pad_mask], dim=1)  # (B, L+1)

        # Bidirectional Transformer (no causal mask)
        encoded    = self.transformer(seq, src_key_padding_mask=src_key_mask)

        cls_output = encoded[:, 0, :]                  # (B, D)
        mlm_logits = self.mlm_head(encoded[:, 1:, :]) if apply_masking else None

        return cls_output, mlm_logits


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ─────────────────────────────────────────────────────────────────────────────
# B. DEEP & CROSS NETWORK v2  (§4.3.2)
# ─────────────────────────────────────────────────────────────────────────────

class CrossLayer(nn.Module):
    """Single DCN-v2 cross layer: x_{l+1} = x_0 · (W_l x_l + b_l) + x_l"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.W  = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.W(xl) + xl


class DCNv2(nn.Module):
    """
    Deep & Cross Network v2.
    Automatically learns explicit high-order feature interactions.
    Example: (is_lunch × is_vegetarian × category=beverage) → high importance.
    """

    def __init__(self, cfg: DCNv2Config):
        super().__init__()
        self.cross_layers = nn.ModuleList(
            [CrossLayer(cfg.input_dim) for _ in range(cfg.cross_layers)]
        )
        deep_layers = []
        prev = cfg.input_dim
        for h in cfg.deep_hidden_dims:
            deep_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = h
        self.deep   = nn.Sequential(*deep_layers)
        self.output = nn.Linear(cfg.input_dim + prev, cfg.input_dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cross path
        x_cross = x
        for layer in self.cross_layers:
            x_cross = layer(x, x_cross)

        # Deep path
        x_deep = self.deep(x)

        # Stack and project
        combined = torch.cat([x_cross, x_deep], dim=-1)
        return F.relu(self.output(combined))


# ─────────────────────────────────────────────────────────────────────────────
# C. MULTI-GATE MIXTURE OF EXPERTS  (§4.3.3)
# ─────────────────────────────────────────────────────────────────────────────

class ExpertNetwork(nn.Module):
    """Single expert: a 2-layer MLP."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MMoE(nn.Module):
    """
    Multi-Gate Mixture of Experts with 2 tasks:
      Task 1: P(Accept)  → sigmoid
      Task 2: E(AOV Lift) → linear regression

    Final score = α × P(Accept) + (1−α) × E(AOV Lift)

    The meal_completeness_score feeds into gating:
      - Low completeness  → routes to P(Accept) expert
      - High completeness → routes to E(AOV Lift) expert
    """

    def __init__(self, cfg: MMoEConfig):
        super().__init__()
        self.cfg = cfg

        # Shared expert pool
        self.experts = nn.ModuleList([
            ExpertNetwork(cfg.input_dim, cfg.expert_hidden_dim, cfg.dropout)
            for _ in range(cfg.num_experts)
        ])

        # Per-task gating networks
        self.gates = nn.ModuleList([
            nn.Linear(cfg.input_dim + 1, cfg.num_experts)  # +1 for completeness signal
            for _ in range(cfg.num_tasks)
        ])

        # Task-specific output heads
        self.accept_head = nn.Linear(cfg.expert_hidden_dim, 1)    # Task 1
        self.aov_head    = nn.Linear(cfg.expert_hidden_dim, 1)    # Task 2

    def forward(
        self,
        x: torch.Tensor,                    # (B, input_dim)
        completeness_score: torch.Tensor,   # (B, 1)  — routes gating
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            p_accept:   (B,) — P(item is accepted)
            aov_lift:   (B,) — Expected AOV lift
        """
        # Compute expert outputs: list of (B, expert_hidden_dim)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, K, H)

        # Gating input: concatenate features + completeness signal
        gate_input = torch.cat([x, completeness_score], dim=-1)         # (B, D+1)

        # Task 1 (P Accept)
        gate1   = F.softmax(self.gates[0](gate_input), dim=-1)          # (B, K)
        mix1    = (gate1.unsqueeze(-1) * expert_outs).sum(dim=1)        # (B, H)
        p_accept = self.accept_head(mix1).squeeze(-1).sigmoid()         # (B,)

        # Task 2 (AOV Lift)
        gate2    = F.softmax(self.gates[1](gate_input), dim=-1)
        mix2     = (gate2.unsqueeze(-1) * expert_outs).sum(dim=1)
        aov_lift = self.aov_head(mix2).squeeze(-1)                       # (B,) — unbounded

        return p_accept, aov_lift

    def get_final_score(
        self,
        p_accept: torch.Tensor,
        aov_lift: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """Score = α × P(Accept) + (1−α) × E(AOV Lift)"""
        a = alpha if alpha is not None else self.cfg.alpha
        # Normalise AOV lift to [0,1] for combination
        aov_norm = torch.sigmoid(aov_lift / 50.0)    # Assuming AOV lift in ₹ scale
        return a * p_accept + (1 - a) * aov_norm


# ─────────────────────────────────────────────────────────────────────────────
# FULL DCTN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DCTN(nn.Module):
    """
    Deep Cross Transformer Network — full ranking model.

    Input:  cart_ids, attention_mask, candidate features, user/context/cross features
    Output: p_accept (B,), aov_lift (B,), final_score (B,)
    """

    def __init__(self, cfg: DCTNConfig):
        super().__init__()
        self.cfg = cfg

        # Component A: Masked Cart Encoder
        self.cart_encoder = MaskedCartEncoder(cfg.cart_encoder)

        # Projection: cart CLS + all features → DCN-v2 input dim
        cart_dim     = cfg.cart_encoder.item_embed_dim
        feature_dim  = (cfg.user_feature_dim + cfg.item_feature_dim +
                        cfg.context_feature_dim + cfg.cross_feature_dim)
        combined_dim = cart_dim + feature_dim

        self.input_projection = nn.Sequential(
            nn.Linear(combined_dim, cfg.dcn.input_dim),
            nn.LayerNorm(cfg.dcn.input_dim),
            nn.ReLU(),
        )

        # Component B: DCN-v2
        self.dcn = DCNv2(cfg.dcn)

        # Component C: MMoE
        dcn_out_dim = cfg.dcn.input_dim // 2
        # Re-configure MMoE input dim to match DCN-v2 output
        mmoe_cfg = cfg.mmoe
        mmoe_cfg.input_dim = dcn_out_dim
        self.mmoe = MMoE(mmoe_cfg)

    def forward(
        self,
        cart_ids: torch.Tensor,              # (B, L)
        attention_mask: torch.Tensor,        # (B, L)
        user_features: torch.Tensor,         # (B, user_dim)
        item_features: torch.Tensor,         # (B, item_dim)
        context_features: torch.Tensor,      # (B, ctx_dim)
        cross_features: torch.Tensor,        # (B, cross_dim)
        completeness_score: torch.Tensor,    # (B, 1)
        apply_masking: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns: p_accept, aov_lift, final_score, mlm_logits (during training)
        """
        # Component A: Cart encoding
        cls_output, mlm_logits = self.cart_encoder(cart_ids, attention_mask, apply_masking)

        # Concatenate all features
        combined = torch.cat([
            cls_output, user_features, item_features, context_features, cross_features
        ], dim=-1)                                   # (B, combined_dim)

        # Project to DCN-v2 input dimension
        projected = self.input_projection(combined)  # (B, dcn_input_dim)

        # Component B: Cross feature interactions
        dcn_out = self.dcn(projected)                # (B, dcn_out_dim)

        # Component C: MMoE scoring
        p_accept, aov_lift = self.mmoe(dcn_out, completeness_score)
        final_score = self.mmoe.get_final_score(p_accept, aov_lift)

        return p_accept, aov_lift, final_score, mlm_logits


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE TRAINING LOSS  (§5.2)
# ─────────────────────────────────────────────────────────────────────────────

class CompositeLoss(nn.Module):
    """
    L_total = L_rank + λ × L_aov

    L_rank: ListNet loss (approximates LambdaRank — optimises NDCG)
    L_aov:  MSE loss on AOV lift regression head
    L_mcm:  Cross-entropy on masked item reconstruction (MCM, during pretraining)
    """

    def __init__(self, lambda_aov: float = 0.3, lambda_mcm: float = 0.1):
        super().__init__()
        self.lambda_aov = lambda_aov
        self.lambda_mcm = lambda_mcm

    def listnet_loss(
        self,
        scores: torch.Tensor,    # (B,) — model ranking scores
        labels: torch.Tensor,    # (B,) — binary relevance labels
    ) -> torch.Tensor:
        """
        ListNet top-1 approximation: minimise KL divergence between
        softmax of scores and softmax of labels.
        Serves as a differentiable proxy for NDCG optimisation.
        """
        # Group by session — here we treat the full batch as one list
        pred_probs  = F.softmax(scores, dim=0)
        label_probs = F.softmax(labels.float(), dim=0)
        loss = -(label_probs * torch.log(pred_probs + 1e-9)).sum()
        return loss

    def forward(
        self,
        final_score: torch.Tensor,      # (B,)
        p_accept: torch.Tensor,         # (B,)
        aov_lift_pred: torch.Tensor,    # (B,)
        labels: torch.Tensor,           # (B,) binary
        aov_lift_true: torch.Tensor,    # (B,) actual AOV lift
        mlm_logits: Optional[torch.Tensor] = None,   # (B, L, V)
        mlm_targets: Optional[torch.Tensor] = None,  # (B, L)
        mcm_mask: Optional[torch.Tensor] = None,     # (B, L) bool
    ) -> Tuple[torch.Tensor, dict]:
        # Ranking loss (ListNet on final score)
        l_rank = self.listnet_loss(final_score, labels)

        # AOV loss (MSE on positive examples only)
        pos_mask = labels.bool()
        if pos_mask.any():
            l_aov = F.mse_loss(
                aov_lift_pred[pos_mask],
                aov_lift_true[pos_mask].clamp(0, 200),  # cap at ₹200
            )
        else:
            l_aov = torch.tensor(0.0, device=labels.device)

        # Binary cross-entropy loss (auxiliary, helps calibrate P(Accept))
        l_bce = F.binary_cross_entropy(p_accept, labels)

        total = l_rank + self.lambda_aov * l_aov + 0.1 * l_bce

        # MCM loss (only during MCM pre-training phase)
        l_mcm = torch.tensor(0.0, device=labels.device)
        if mlm_logits is not None and mlm_targets is not None and mcm_mask is not None:
            masked_logits  = mlm_logits[mcm_mask]
            masked_targets = mlm_targets[mcm_mask]
            if masked_logits.shape[0] > 0:
                l_mcm  = F.cross_entropy(masked_logits, masked_targets)
                total  = total + self.lambda_mcm * l_mcm

        breakdown = {
            "loss_total": total.item(),
            "loss_rank":  l_rank.item(),
            "loss_aov":   l_aov.item(),
            "loss_bce":   l_bce.item(),
            "loss_mcm":   l_mcm.item(),
        }
        return total, breakdown
