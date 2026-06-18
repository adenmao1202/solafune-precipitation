import segmentation_models_pytorch as smp
import torch.nn as nn
from dataset import IN_CHANNELS


def build_model(encoder_name: str = "efficientnet-b4",
                encoder_weights=None) -> nn.Module:
    """
    UNet with EfficientNet-B4 encoder。

    encoder_weights=None：從零訓練。
    四篇同領域論文（GENESIS/NPM/TUPANN/GlobalMetNet）均從零訓練，不用 ImageNet。
    原因：51ch 多光譜輸入與 RGB 自然影像差距太大，SMP 只給前 3ch 用 ImageNet 權重，
    其餘 48ch 隨機初始化，造成 gradient 更新嚴重不平衡。

    輸入：IN_CHANNELS channels（多時序 + mask）
    輸出：1 channel（log1p 降水量）
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=IN_CHANNELS,
        classes=1,
        activation=None,
        decoder_use_batchnorm=True,
        decoder_attention_type=None,
    )
    # 在 decoder 的每個 block 加 dropout 0.2，對抗 overfitting
    for block in model.decoder.blocks:
        block.conv1 = nn.Sequential(block.conv1, nn.Dropout2d(p=0.2))

    return model
