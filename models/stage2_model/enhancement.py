import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stage2_model.parts import ResDenseBlock


def padding_concat(x, y):
    x_size = x.shape[2:]  # w*h
    y_size = y.shape[2:]

    # 如果两个的形状不同，使用0进行填充
    w_dif = x_size[0] - y_size[0]
    h_dif = x_size[1] - y_size[1]
    if w_dif + h_dif != 0:
        if w_dif > 0:
            y = F.pad(y, (0, 0, 0, w_dif), mode='constant', value=0)

        elif w_dif < 0:
            x = F.pad(x, (0, 0, 0, -w_dif), mode='constant', value=0)

        if h_dif > 0:
            y = F.pad(y, (0, h_dif, 0, 0), mode='constant', value=0)  # [left, right, top, bot]
        elif h_dif < 0:
            x = F.pad(x, (0, -h_dif, 0, 0), mode='constant', value=0)

    return torch.cat((x, y), dim=1)


def down_sampling(in_channels, out_channels):
    """
        下采样：每次下采样后Feature Map的尺寸减半，通道数乘2
    """
    module = nn.Sequential(
        # nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=1),
        # @CHANGE: IN -> BN
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )
    return module


def up_sampling(in_channels, out_channels):
    """
        上采样：转置卷积上采样，每次上采样后Feature Map的尺寸乘2
    """
    module = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        # @CHANGE: IN -> BN
        # nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )
    return module


# Encoder
# @change: 修改网络层数从3,6,8,8到3,3,6,6
# @CHANGE: growth ratio = 32 -> 64
class Enhancement_Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Enhancement_Encoder, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        self.block1 = ResDenseBlock(3, 64, 64, 64)  # 64, 256, 256
        self.down1 = down_sampling(64, 128)  # 128, 128

        self.block2 = ResDenseBlock(3, 128, 128, 64)  # 128, 128, 128
        self.down2 = down_sampling(128, 256)  # 64, 64

        self.block3 = ResDenseBlock(3, 256, 256, 64, is_IN=True)  # 256, 64, 64
        self.down3 = down_sampling(256, 512)  # 32, 32

        self.block4 = ResDenseBlock(3, 512, 512, 64, is_IN=True)  # 512, 32, 32
        self.down4 = down_sampling(512, 1024)  # 16, 16

        self.battle_neck = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )  # 1024, 16, 16

    def forward(self, x):
        """
            x is the input image
        """
        x1 = self.block1(self.in_conv(x))
        x2 = self.block2(self.down1(x1))
        x3 = self.block3(self.down2(x2))
        x4 = self.block4(self.down3(x3))
        x_battle_neck = self.battle_neck(self.down4(x4))
        embedding = x1  # 用于迁移的风格向量
        enc_outs = (x1, x2, x3, x4, x_battle_neck)  # 用于恢复图像的编码结果
        # @change: 可以把Embedding直接替换为x1, 后续的训练代码不需要改变
        return embedding, enc_outs


# Decoder
# @change: 修改网络层数从3,6,8,8到3,3,4,4
class Enhancement_Decoder(nn.Module):
    def __init__(self):
        super(Enhancement_Decoder, self).__init__()

        self.up1 = up_sampling(1024, 512)  # 512, 32, 32
        self.block1 = ResDenseBlock(3, 1024, 1024, 64, False)  # 512, 32, 32

        self.up2 = up_sampling(1024, 256)  # 256, 64, 64
        self.block2 = ResDenseBlock(3, 512, 512, 64, False)  # 512, 64, 64

        self.up3 = up_sampling(512, 128)  # 128, 128, 128
        self.block3 = ResDenseBlock(3, 256, 256, 64, False)  # 256, 128, 128

        self.up4 = up_sampling(256, 64)  # 64, 256, 256
        self.block4 = ResDenseBlock(3, 128, 128, 64, False)  # 128, 256, 256

        self.out_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, enc_outs):
        x = self.block1(padding_concat(self.up1(enc_outs[-1]), enc_outs[-2]))
        x = self.block2(padding_concat(self.up2(x), enc_outs[-3]))
        x = self.block3(padding_concat(self.up3(x), enc_outs[-4]))
        x = self.block4(padding_concat(self.up4(x), enc_outs[-5]))
        x = self.out_conv(x)
        return x


def test():
    encoder = Enhancement_Encoder()
    decoder = Enhancement_Decoder()

    x = torch.randn(1, 3, 1014, 514)
    embedding, enc_outs = encoder(x)
    generated = decoder(enc_outs)
    print(generated.shape)


if __name__ == "__main__":
    test()
