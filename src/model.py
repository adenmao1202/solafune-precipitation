import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from dataset import IN_CHANNELS


class FiLMLayer(nn.Module):
    """Modulates bottleneck features with time encoding (day + hour sin/cos).
    Zero-initialized so it acts as identity at the start of training."""
    def __init__(self, time_dim: int, feature_channels: int):
        super().__init__()
        self.gamma_fc = nn.Linear(time_dim, feature_channels)
        self.beta_fc  = nn.Linear(time_dim, feature_channels)
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, features: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(time_feat)[:, :, None, None]
        beta  = self.beta_fc(time_feat)[:, :, None, None]
        return features * (1 + gamma) + beta


class PrecipUNet(smp.Unet):
    def __init__(self, time_dim: int = 4, **kwargs):
        super().__init__(**kwargs)
        bottleneck_ch = self.encoder.out_channels[-1]
        self.film = FiLMLayer(time_dim=time_dim, feature_channels=bottleneck_ch)
        for block in self.decoder.blocks:
            block.conv1 = nn.Sequential(block.conv1, nn.Dropout2d(p=0.2))

    def forward(self, x: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        features = list(self.encoder(x))
        features[-1] = self.film(features[-1], time_feat)
        decoder_output = self.decoder(features)
        return self.segmentation_head(decoder_output)


def build_model(encoder_name: str = "efficientnet-b4",
                encoder_weights: str = "imagenet") -> nn.Module:
    """
    UNet with EfficientNet-B4 encoder + FiLM time conditioning.

    FiLM injects day-of-year and hour-of-day (sin/cos) into the bottleneck,
    allowing the model to learn seasonal and diurnal precipitation patterns.
    Based on NPM paper ablation: day encoding alone = +17% CSI.

    encoder_weights="imagenet": front 3ch get ImageNet prior (VIS/NIR channels).
    v7 showed Kaiming reinit is worse than partial ImageNet for our 51ch input.
    """
    model = PrecipUNet(
        time_dim=4,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=IN_CHANNELS,
        classes=1,
        activation=None,
        decoder_use_batchnorm=True,
        decoder_attention_type=None,
    )

    return model
