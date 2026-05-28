from transformers import PretrainedConfig


class RADARConfig(PretrainedConfig):
    """HuggingFace-compatible configuration for the RADAR model."""

    model_type = "radar"

    def __init__(
        self,
        base_model_name: str = "answerdotai/ModernBERT-large",
        style_dim: int = 38,
        style_proj_dim: int = 128,
        fusion_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):

        self.base_model_name = base_model_name
        self.style_dim = style_dim
        self.style_proj_dim = style_proj_dim
        self.fusion_dim = fusion_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        super().__init__(**kwargs)
