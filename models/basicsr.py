from basicsr.models.archs.discriminator import MultiscaleDiscriminator
from basicsr.models.archs.msdi2e_arch import MSDI2E
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

class NoConfig(PretrainedConfig):
    model_type = "any"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MSDI2EConfig(PretrainedConfig):
    model_type = "MSDI2E"

    def __init__(self, depth=5, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth


class MSDI2EModel(PreTrainedModel):
    config_class = MSDI2EConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MSDI2E(depth=config.depth)
        self.init_weights()

    def forward(self, x, y=None):
        if self.training:
            return self.model(x, y)
        else:
            return self.forward_eval(x)

    def forward_eval(self, x):
        prior_z = self.model.net_prior(x)
        latent_list_inverse = self.model.prior_upsampling(prior_z)
        out = self.model.generator(x, latent_list_inverse)
        return out


class MultiscaleDiscriminatorModel(PreTrainedModel):
    config_class = NoConfig

    def __init__(self):
        super().__init__(NoConfig())
        self.model = MultiscaleDiscriminator()
        self.init_weights()

    def forward(self, x):
        return self.model(x)
