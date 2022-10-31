import torch
import torch.nn as nn


# def conv(input_channels, output_channels, is_down, layer_norm=False):
#     """
#         上采样/下采样过程中的卷积块，上采样过程中不使用批量正则化
#     """
#     if is_down is True:
#         return nn.Sequential(
#             nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1),
#             # @change: 在4.5版本中，正则化方法就已经从Batch_Norm改为了InstanceNorm
#             nn.InstanceNorm2d(output_channels) if not layer_norm else nn.GroupNorm(1, output_channels),
#             nn.ReLU(),
#         )
#     elif is_down is False:
#         return nn.Sequential(
#             nn.ConvTranspose2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1)),
#             nn.ReLU(),
#         )

def conv(input_channels, output_channels, is_down=False, is_IN=False):
    """
        上采样/下采样过程中的卷积块，上采样过程中不使用批量正则化

    @change:
        编码和解码过程统一采用Instance_Norm，取消编码过程对layer_norm的使用；
        修正解码过程的中的转置卷积为卷积；
        修改编码阶段使用LeakyReLu(0.2), 解码阶段使用ReLU；
        UGAN解码阶段没有使用正则化，这里使用Instance_Norm
    """
    if is_down is True:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1),
            # @CHANGE: IN -> BN
            nn.BatchNorm2d(output_channels) if not is_IN else nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2),
        )
    elif is_down is False:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1)),
            # nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2),
        )


class ResDenseBlock(nn.Module):
    def __init__(self, num_layers, input_channels, output_channels,
                 growth_ratio, is_down=True, is_IN=False):
        """
            残差密集块，一个残差密集块包含多个密集连接的卷积层和位于最后的过渡层，
            过渡层调整最终输出的通道数到指定的大小并与原始的输入进行残差连接
            :param num_layers: 密集块的卷积层数
            :param input_channels: 输入通道数
            :param output_channels: 输出通道数，输出通道数必须等于输入通道数
            :param growth_ratio: 增长率
            :param is_down: 采用卷积还是转置卷积
        """
        super(ResDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.is_down = is_down
        self.is_IN = is_IN
        assert input_channels == output_channels, \
            "The input_channels and output_channels of ResDenseBlock are supposed to match"

        layer = []
        for i in range(num_layers):
            layer.append(
                conv(input_channels + growth_ratio * i, growth_ratio, is_down, is_IN),
            )
        self.dense = nn.Sequential(*layer)

        self.transition = nn.Conv2d(input_channels + growth_ratio * num_layers,
                                    output_channels,
                                    kernel_size=(1, 1),
                                    stride=(1, 1)
                                    )

    def forward(self, x):
        ori_input = x
        for i, blk in enumerate(self.dense):
            y = blk(x)
            # 连接通道维度上每个块的输入和输出
            x = torch.cat((x, y), dim=1)
        x = self.transition(x)
        return x + ori_input
