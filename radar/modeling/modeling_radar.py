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
    logits: torch.FloatTensor = None
    h_fused: Optional[torch.FloatTensor] = None  # exposed for auxiliary losses


class RADARModel(PreTrainedModel):
    """
    Full RADAR detection model.

    Components:
      1. ARSE: ModernBERT-large encoder → h_semantic (hidden_size)
      2. SLPE: Learnable projection of 38 stylometric features → h_style (128)
      3. CrossAttentionFusion: h_semantic × h_style → h_fused (512)


    Forward returns a dict with keys:
      - logits    raw logit from primary head
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
          - h_fused     fused representation (exposed for auxiliary losses)
        """
        h_fused = self.encode(input_ids, attention_mask, style_features)

        # Primary classification logit
        logit = self.classifier(h_fused).squeeze(-1)  # [B]

        return RADAROutput(
            logits=logit,
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


