#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Embedding Layers（嵌入层 - Input Embedding 的核心实现）
#
#################################################################################################################

import torch
import torch.nn as nn


class ResUnit(nn.Module):
    """
    残差单元
    基础构建块，结合卷积、激活函数和批归一化，带有残差连接
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 8,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=k,
                dilation=dilation,  # 扩张率，控制感受野大小
                stride=stride,
                bias=bias,
                padding="same",
            ),
            nn.GELU(),            # 激活函数
            nn.BatchNorm1d(c_out), # 批归一化
        )
        # 如果输入输出通道数不同，需要匹配维度
        if c_in > 1 and c_in != c_out:
            self.match_residual = True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual = False

    def forward(self, x) -> torch.Tensor:
        if self.match_residual:
            x_bottleneck = self.conv(x)  # 调整残差维度
            x = self.layers(x)
            return torch.add(x_bottleneck, x)  # 残差连接
        else:
            return torch.add(x, self.layers(x))  # 残差连接

# 原始输入 → 实例归一化 → 【扩张卷积DilatedBlock】 → 时间特征投影 → 拼接 → Transformer编码器
class DilatedBlock(nn.Module):
    """
    扩张卷积块 - NILMFormer 的 Input Embedding 核心组件

    通过堆叠多个不同扩张率的残差单元，提取多尺度时序特征
    这是 NILMFormer 的关键创新之一：用扩张卷积替代传统的位置编码

    示例：dilation_list=[1, 2, 4, 8]
    - dilation=1: 感受野 = k (局部细节特征)
    - dilation=2: 感受野 = 2k (中等范围特征)
    - dilation=4: 感受野 = 4k (较长范围依赖)
    - dilation=8: 感受野 = 8k (长期时序模式)
    """
    def __init__(
        self,
        c_in: int = 32,
        c_out: int = 32,
        kernel_size: int = 8,
        # 使用不同扩张率的残差单元堆叠
        # dilation=1: 局部特征
        # dilation=2,4,8: 逐渐增大的感受野
        dilation_list: list = [1, 2, 4, 8],  # 扩张率配置
        bias: bool = True,
    ):
        super().__init__()

        layers = []
        for i, dilation in enumerate(dilation_list):
            if i == 0:
                # 第一层：输入通道 c_in -> 输出通道 c_out
                layers.append(
                    ResUnit(c_in, c_out, k=kernel_size, dilation=dilation, bias=bias)
                )
            else:
                # 后续层：保持 c_out 维度
                layers.append(
                    ResUnit(c_out, c_out, k=kernel_size, dilation=dilation, bias=bias)
                )
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        """
        输入: (B, c_in, L) - 原始负荷曲线
        输出: (B, c_out, L) - 多尺度嵌入特征
        """
        return self.network(x)
