import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from dataset import IN_CHANNELS_12, IN_CHANNELS_12DIFF, COND_DIM


class FiLMLayer(nn.Module):
    """Modulates bottleneck features with time+satellite conditioning.
    Zero-initialized so it acts as identity at the start of training."""
    def __init__(self, cond_dim: int, feature_channels: int):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_channels)
        self.beta_fc  = nn.Linear(cond_dim, feature_channels)
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, features: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(cond)[:, :, None, None]
        beta  = self.beta_fc(cond)[:, :, None, None]
        return features * (1 + gamma) + beta


class PrecipUNet(smp.Unet):
    def __init__(self, cond_dim: int = COND_DIM, use_dual_head: bool = False, **kwargs):
        super().__init__(**kwargs)
        bottleneck_ch = self.encoder.out_channels[-1]
        self.film = FiLMLayer(cond_dim=cond_dim, feature_channels=bottleneck_ch)
        for block in self.decoder.blocks:
            block.conv1 = nn.Sequential(block.conv1, nn.Dropout2d(p=0.2))
        self.use_dual_head = use_dual_head
        if use_dual_head:
            # decoder_channels[-1] is the output channel count of the last decoder block
            dec_out_ch = self.decoder.blocks[-1].conv2[0].out_channels
            self.rain_head = nn.Sequential(
                nn.Conv2d(dec_out_ch, 1, kernel_size=1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        features = list(self.encoder(x))
        features[-1] = self.film(features[-1], cond)
        decoder_output = self.decoder(features)
        intensity = self.segmentation_head(decoder_output)
        if self.use_dual_head:
            rain = self.rain_head(decoder_output)
            return {"intensity": intensity, "rain": rain}
        return intensity


def build_model(encoder_name: str = "efficientnet-b4",
                encoder_weights: str = "imagenet",
                num_classes: int = 1,
                in_channels: int | None = None,
                cond_dim: int = COND_DIM,
                use_dual_head: bool = False) -> nn.Module:
    """
    UNet with EfficientNet-B4 encoder + FiLM conditioning.

    FiLM injects [sin/cos day, sin/cos hour, satellite one-hot (3)] into the
    bottleneck, letting the model learn seasonal, diurnal, and per-satellite
    precipitation patterns.

    in_channels: defaults to IN_CHANNELS_12 (39ch). Pass IN_CHANNELS_18 (57ch)
                 for 18-slot canonical mapping.
    cond_dim: defaults to COND_DIM (7). Must match dataset output.
    """
    if in_channels is None:
        in_channels = IN_CHANNELS_12
    model = PrecipUNet(
        cond_dim=cond_dim,
        use_dual_head=use_dual_head,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
        decoder_use_batchnorm=True,
        decoder_attention_type=None,
    )
    return model
