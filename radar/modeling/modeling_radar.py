"""
RADAR: Robust Adversarial-Resistant Detection with Adaptive Reasoning

Architecture:
  ARSE   - Adversarially Robust Semantic Encoder (ModernBERT-large)
  SLPE   - Statistical Linguistic Profile Extractor (38 stylometric features)
  UACC   - Uncertainty-Aware Calibrated Classifier (dual-head, temperature scaling)

Fusion via cross-attention + MLP following the research proposal.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_radar import RADARConfig

# ---------------------------------------------------------------------------
# Cross-Attention Fusion Layer
# ---------------------------------------------------------------------------


class CrossAttentionFusion(torch.nn.Module):
    """
    Fuses semantic (ARSE) and stylometric (SLPE) representations.

    The semantic embedding attends to the projected style embedding,
    then the attended output is concatenated with the original style
    features and passed through a projection MLP.
    """

    def __init__(
        self,
        semantic_dim: int,
        style_dim: int,
        style_proj_dim: int,
        fusion_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project style features to semantic_dim for attention compatibility
        self.style_to_semantic = torch.nn.Linear(style_proj_dim, semantic_dim)

        # Multi-head cross-attention: query = semantic, key/value = projected style
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = torch.nn.LayerNorm(semantic_dim)

        # Fusion MLP: concat attended semantic + raw style projection → fusion_dim
        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(semantic_dim + style_proj_dim, fusion_dim),
            torch.nn.LayerNorm(fusion_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

    def forward(
        self, h_semantic: torch.Tensor, h_style_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_semantic   : [B, semantic_dim] CLS embedding from ARSE
        h_style_proj : [B, style_proj_dim] projected stylometric features

        Returns
        -------
        h_fused : [B, fusion_dim]
        """
        # Project style to match semantic dimension for attention
        style_as_semantic = self.style_to_semantic(h_style_proj)  # [B, semantic_dim]

        # Add sequence dimension: [B, 1, dim]
        query = h_semantic.unsqueeze(1)
        kv = style_as_semantic.unsqueeze(1)

        # Cross-attention: semantic query attends to style key/value
        attended, _ = self.cross_attn(query, kv, kv)
        attended = attended.squeeze(1)  # [B, semantic_dim]

        # Residual connection + norm
        attended = self.attn_norm(attended + h_semantic)

        # Concatenate attended semantic + original style projection
        h_concat = torch.cat(
            [attended, h_style_proj], dim=-1
        )  # [B, semantic_dim + style_proj_dim]

        # Project to fusion dimension
        h_fused = self.fusion_mlp(h_concat)  # [B, fusion_dim]
        return h_fused


# ---------------------------------------------------------------------------
# RADAR Model
# ---------------------------------------------------------------------------


@dataclass
class RADAROutput(ModelOutput):
    """
    Output dataclass for RADARModel forward pass.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    score: torch.FloatTensor = None
    uncertainty: torch.FloatTensor = None
    h_fused: Optional[torch.FloatTensor] = None  # exposed for auxiliary losses


class RADARModel(PreTrainedModel):
    """
    Full RADAR detection model.

    Components:
      1. ARSE: ModernBERT-large encoder → h_semantic (hidden_size)
      2. SLPE: Learnable projection of 38 stylometric features → h_style (128)
      3. CrossAttentionFusion: h_semantic × h_style → h_fused (512)
      4. UACC: Primary classifier + Uncertainty head → calibrated score

    Forward returns a dict with keys:
      - loss      (if labels provided)
      - logits    raw logit from primary head
      - score     calibrated score ∈ [0, 1]
      - uncertainty  predicted uncertainty ∈ [0, 1]
    """

    config_class = RADARConfig
    all_tied_weights_keys = OrderedDict()
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True

    def __init__(self, config: RADARConfig):
        super().__init__(config)

        # --- ARSE: Semantic Encoder ---
        encoder_config = AutoConfig.from_pretrained(config.base_model_name)
        self.encoder = AutoModel.from_config(encoder_config)
        hidden_size = encoder_config.hidden_size

        # --- SLPE: Stylometric Feature Projection ---
        self.style_proj = nn.Sequential(
            nn.Linear(config.style_dim, config.style_proj_dim),
            nn.LayerNorm(config.style_proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # --- Cross-Attention Fusion ---
        self.fusion = CrossAttentionFusion(
            semantic_dim=hidden_size,
            style_dim=config.style_dim,
            style_proj_dim=config.style_proj_dim,
            fusion_dim=config.fusion_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

        # --- UACC: Dual-Head Classifier ---
        self.classifier = nn.Linear(config.fusion_dim, 1)
        self.uncertainty_head = nn.Linear(config.fusion_dim, 1)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_encoder(self) -> nn.Module:
        return self.encoder

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text (and optionally style features) into a fused representation.

        Returns h_fused : [B, fusion_dim]
        """
        # Run transformer encoder
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use CLS token representation
        h_semantic = encoder_output.last_hidden_state[:, 0, :]  # [B, hidden_size]

        if style_features is not None:
            # Project stylometric features
            h_style_proj = self.style_proj(style_features)  # [B, style_proj_dim]
            # Fuse
            h_fused = self.fusion(h_semantic, h_style_proj)  # [B, fusion_dim]
        else:
            # If no style features provided, project semantic directly
            B, H = h_semantic.shape
            dummy_style = h_semantic.new_zeros(B, self.config.style_proj_dim)
            h_fused = self.fusion(h_semantic, dummy_style)

        return h_fused

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> RADAROutput:
        """
        Forward pass.

        Parameters
        ----------
        input_ids      : [B, seq_len]
        attention_mask : [B, seq_len]
        style_features : [B, 38]  optional stylometric features
        labels         : [B]  float labels 0.0 / 1.0 for loss computation

        Returns
        -------
        RADAROutput with fields:
          - loss        (if labels provided) weighted BCE on calibrated score
          - logits      raw logit from primary head (before sigmoid)
          - score       calibrated probability score
          - uncertainty uncertainty estimate from dual-head
          - h_fused     fused representation (exposed for auxiliary losses)
        """
        h_fused = self.encode(input_ids, attention_mask, style_features)

        # Primary classification logit
        logit = self.classifier(h_fused).squeeze(-1)  # [B]

        # Uncertainty estimate
        uncertainty = torch.sigmoid(self.uncertainty_head(h_fused)).squeeze(-1)  # [B]

        # Calibrated probability
        p = torch.sigmoid(logit / self.config.temperature)  # [B]

        # UACC score: pulls towards 0.5 when uncertainty is high
        score = 0.5 + (p - 0.5) * (1.0 - uncertainty)  # [B]

        loss = None
        if labels is not None:
            # Primary loss on calibrated score (BCE)
            loss = nn.BCELoss()(score, labels.float())

        return RADAROutput(
            loss=loss,
            logits=logit,
            score=score,
            uncertainty=uncertainty,
            h_fused=h_fused,  # exposed for AIT triplet / invariance losses
        )

    @classmethod
    def from_pretrained_encoder(
        cls, encoder_name: str = "answerdotai/ModernBERT-large", **kwargs
    ) -> "RADARModel":
        """
        Create a RADAR model and load pretrained encoder weights.
        All other weights are randomly initialized.
        """
        config = RADARConfig(base_model_name=encoder_name, **kwargs)
        model = cls(config)
        encoder = AutoModel.from_pretrained(encoder_name)
        model.encoder.load_state_dict(encoder.state_dict(), strict=False)
        return model

    def set_temperature(self, temperature: float) -> None:
        """Update the calibration temperature (used after temperature scaling)."""
        self.config.temperature = temperature

    def set_c_at_1_threshold(self, threshold: float) -> None:
        """Update the C@1 abstention threshold."""
        self.config.c_at_1_threshold = threshold

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_features: Optional[torch.Tensor] = None,
        apply_c_at_1: bool = True,
    ) -> np.ndarray:
        """
        Run inference and return scores as a numpy array.
        Scores of exactly 0.5 represent abstention (used in C@1).

        Parameters
        ----------
        apply_c_at_1 : if True, scores within c_at_1_threshold of 0.5 are
                       snapped to 0.5 to optimize the C@1 metric.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, style_features)
            scores = outputs.score.cpu().numpy()

        if apply_c_at_1:
            threshold = self.config.c_at_1_threshold
            scores[np.abs(scores - 0.5) < threshold] = 0.5

        return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Adversarial Invariance Loss
# ---------------------------------------------------------------------------


class AdversarialInvarianceLoss(nn.Module):
    """
    L_inv = MSE(encode(x), encode(x_tilde))

    Encourages the model to produce similar representations for a text
    and its adversarially obfuscated version.
    """

    def forward(
        self,
        h_original: torch.Tensor,
        h_augmented: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.mse_loss(h_original, h_augmented)


# ---------------------------------------------------------------------------
# Contrastive Triplet Loss
# ---------------------------------------------------------------------------


class TripletLoss(nn.Module):
    """
    Triplet margin loss over (anchor_human, positive_human, negative_AI).

    anchor   : human text embedding
    positive : another human text embedding
    negative : AI text embedding

    L_triplet = max(0, ||f(h) - f(h+)||² - ||f(h) - f(n)||² + margin)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.triplet = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return self.triplet(anchor, positive, negative)


# ---------------------------------------------------------------------------
# Multi-Task Adversarial Contrastive Loss (MACL)
# ---------------------------------------------------------------------------


class MACLoss(nn.Module):
    """
    Combined training objective for Radar model:

      L = α · L_cls + β · L_inv + γ · L_triplet

    where
      L_cls     : weighted BCE on calibrated scores
      L_inv     : adversarial invariance MSE (when adversarial pairs provided)
      L_triplet : contrastive triplet loss (when human/AI triplets provided)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        pos_weight: Optional[float] = None,
        margin: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if pos_weight is not None:
            pw = torch.tensor([pos_weight])
            self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.cls_loss = nn.BCEWithLogitsLoss()

        self.inv_loss = AdversarialInvarianceLoss()
        self.triplet_loss = TripletLoss(margin=margin)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        h_original: Optional[torch.Tensor] = None,
        h_augmented: Optional[torch.Tensor] = None,
        h_anchor: Optional[torch.Tensor] = None,
        h_positive: Optional[torch.Tensor] = None,
        h_negative: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Classification loss
        loss = self.alpha * self.cls_loss(logits, labels.float())

        # Adversarial invariance loss (if adversarial pairs provided)
        if h_original is not None and h_augmented is not None:
            loss = loss + self.beta * self.inv_loss(h_original, h_augmented)

        # Contrastive triplet loss (if triplets provided)
        if h_anchor is not None and h_positive is not None and h_negative is not None:
            if h_anchor.size(0) > 0:
                loss = loss + self.gamma * self.triplet_loss(
                    h_anchor, h_positive, h_negative
                )

        return loss
