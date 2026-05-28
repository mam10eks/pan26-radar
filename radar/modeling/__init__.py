from .configuration_radar import RADARConfig
from .modeling_radar import (
    CrossAttentionFusion,
    RADARModel,
    RADAROutput,
)

__all__ = [
    "RADARConfig",
    "RADARModel",
    "CrossAttentionFusion",
    "RADAROutput",
    "registers",
]


def registers():
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
    )

    AutoConfig.register("radar", RADARConfig)
    AutoModel.register(RADARConfig, RADARModel)
    AutoModelForSequenceClassification.register(RADARConfig, RADARModel)

    RADARConfig.register_for_auto_class("AutoConfig")
    RADARModel.register_for_auto_class("AutoModel")
    RADARModel.register_for_auto_class("AutoModelForSequenceClassification")
