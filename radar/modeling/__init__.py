from .configuration_radar import RADARConfig
from .modeling_radar import (
    AdversarialInvarianceLoss,
    CrossAttentionFusion,
    MACLoss,
    RADARModel,
    RADAROutput,
    TripletLoss,
)

__all__ = [
    "RADARConfig",
    "RADARModel",
    "CrossAttentionFusion",
    "RADAROutput",
    "AdversarialInvarianceLoss",
    "TripletLoss",
    "MACLoss",
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
