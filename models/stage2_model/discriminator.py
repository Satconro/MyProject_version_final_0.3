import torch
import torch.nn as nn


# Discriminator
def block(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, **kwargs),
        # nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )


class PatchGAN(nn.Module):
    def __init__(self, in_channels=3, features=None):
        super(PatchGAN, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.network = nn.Sequential(
            # 输入图像大小为：3*256*256
            block(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=True),
            # 64*128*128
            block(features[0], features[1], kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=True),
            # 128*64*64
            block(features[1], features[2], kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=True),
            # 256*32*32
            block(features[2], features[3], kernel_size=4, stride=1, padding=1, padding_mode="reflect", bias=True),
            # 512*31*31

            # The last layer: 最后一层不进行实例正则化，使用Sigmoid函数将输入转换为[0-1]的概率分布
            # 最终输出为 1*30*30的概率张量矩阵，每一个坐标代表原图中一片70*70的区域
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=1, padding_mode="reflect"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


# class Enhancement_Discriminator(nn.Module):
#     def __init__(self, in_channels=512):
#         super(Enhancement_Discriminator, self).__init__()
#         self.network = nn.Sequential(
#             # 输入图像大小为：1024*16*16
#             ResDenseBlock(3, in_channels, in_channels, 64, lay_norm=True),
#             self.conv(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
#             ResDenseBlock(3, in_channels // 2, in_channels // 2, 64, lay_norm=True),
#             self.conv(in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
#             ResDenseBlock(3, in_channels // 4, in_channels // 4, 64, lay_norm=True),
#             self.conv(in_channels // 4, 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
#         )
#
#     def conv(self, in_channels, out_channels, **kwargs):
#         return nn.Sequential(
#             # 因为WGAN-GP会对每个输入的鉴别器梯度范数进行单独惩罚，而批量标准化将使其无效。所以图像转换部分的图片鉴别器不使用批量正则化
#             nn.Conv2d(in_channels, out_channels, **kwargs),
#             nn.GroupNorm(1, out_channels),
#             nn.ReLU(),
#         )
#
#     def forward(self, x):
#         return self.network(x)