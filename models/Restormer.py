import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from Restormer.models.archs.restormer_arch import Restormer

class RestormerConfig(PretrainedConfig):
    model_type = "Restormer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

class RestormerModel(PreTrainedModel):
    config_class = RestormerConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Restormer(**config.to_dict())
        self.init_weights()

    def forward(self, x):
        return self.model(x)
