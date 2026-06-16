import segmentation_models_pytorch as smp
import torch.nn as nn
from dataset import IN_CHANNELS


def build_model(encoder_name: str = "efficientnet-b4",
                encoder_weights: str = None) -> nn.Module:
    """
    UNet with EfficientNet-B4 encoder。

    encoder_weights=None：從零訓練（本比賽禁止外部資料，
    但預訓練 ImageNet 權重在比賽規則下屬於允許的開源模型，
    可視需要改為 'imagenet'）。

    輸入：IN_CHANNELS channels（多時序 + mask）
    輸出：1 channel（log1p 降水量）
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=IN_CHANNELS,
        classes=1,
        activation=None,       # 輸出原始值，loss 函數裡再處理
    )
    return model
