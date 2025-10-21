#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Transformer Layers
#
#################################################################################################################

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops

from typing import Any

from src.nilmformer.congif import NILMFormerConfig


class DiagonalMaskFromSeqlen:
    """
    对角线掩码生成器
    用于在自注意力中屏蔽对角线位置（防止时间步自己关注自己）
    """

    def __init__(self, B, L, device="cpu"):
        """
        Args:
            B: batch size
            L: sequence length (序列长度)
            device: 设备类型，默认 cpu（实际调用时会自动传入 GPU 设备）
        """
        with torch.no_grad():
            # 创建对角线掩码：True 表示该位置需要被屏蔽，不让时间步t关注自己，迫使模型必须从其他时间步学习信息
            # shape: (B, 1, L, L)
            self._mask = torch.diag(
                torch.ones(L, dtype=torch.bool, device=device)
            ).repeat(B, 1, 1, 1)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


# 带对角掩码的自注意力
class DiagonnalyMaskedSelfAttention(nn.Module):
    """
    带对角线掩码的自注意力层
    NILMFormer 的核心创新：防止每个时间步关注自己，强制模型学习时序依赖
    """

    def __init__(
        self,
        dim: int,                  # 输入维度（d_model）
        n_heads: int,              # 注意力头数
        head_dim: int,             # 每个头的维度
        dropout: float,            # dropout 比率
        use_efficient_attention: bool = False,  # 是否使用 xformers 优化
    ):
        super().__init__()

        self.use_efficient_attention: bool = use_efficient_attention

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.dropout: float = dropout

        self.scale = head_dim**-0.5  # 缩放因子 1/√d_k

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Q、K、V 投影层（无偏置）
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)

        # 输出投影层
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

# 多头注意力的具体实现
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, seqlen, _ = x.shape

        # Q、K、V的线性投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(batch, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(batch, seqlen, self.n_heads, self.head_dim)

        # 对角掩码应用
        diag_mask = DiagonalMaskFromSeqlen(batch, seqlen, device=xq.device)

        if self.use_efficient_attention:
            output = xops.memory_efficient_attention(
                xq,
                xk,
                xv,
                attn_bias=(diag_mask.mask).repeat(1, self.n_heads, 1, 1).float(),
                p=self.dropout,
            )
        else:
            scale = 1.0 / xq.shape[-1] ** 0.5
            scores = torch.einsum("blhe,bshe->bhls", xq, xk)
            attn = self.attn_dropout(
                torch.softmax(
                    # 屏蔽位置设为负无穷
                    scale * scores.masked_fill_(diag_mask.mask, -np.inf), dim=-1
                )
            )
            output = torch.einsum("bhls,bshd->blhd", attn, xv)

        # 输出投影和dropout
        return self.out_dropout(self.wo(output.reshape(batch, seqlen, -1)))



class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dp_rate: float = 0.0,
        activation: Any = F.gelu,
        bias1: bool = True,
        bias2: bool = True,
    ):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias1)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias2)
        self.dropout = nn.Dropout(dp_rate)
        self.activation = activation

    # 前馈网络
    def forward(self, x) -> torch.Tensor:
        x = self.layer2(self.dropout(self.activation(self.layer1(x))))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, NFconfig: NILMFormerConfig):
        super().__init__()
        assert not NFconfig.d_model % NFconfig.n_head, (
            f"d_model ({NFconfig.d_model}) must be divisible by n_heads ({NFconfig.n_head})"
        )

        # Encoder中使用带掩码的自注意力
        self.attention_layer = DiagonnalyMaskedSelfAttention(
            dim=NFconfig.d_model,
            n_heads=NFconfig.n_head,
            head_dim=NFconfig.d_model // NFconfig.n_head,
            dropout=NFconfig.dp_rate,
            use_efficient_attention=NFconfig.use_efficient_attention,
        )

        self.norm1 = nn.LayerNorm(NFconfig.d_model, eps=NFconfig.norm_eps)
        self.norm2 = nn.LayerNorm(NFconfig.d_model, eps=NFconfig.norm_eps)

        self.dropout = nn.Dropout(NFconfig.dp_rate)

        self.pffn = PositionWiseFeedForward(
            dim=NFconfig.d_model,
            hidden_dim=NFconfig.d_model * NFconfig.pffn_ratio,
            dp_rate=NFconfig.dp_rate,
        )


    # 识别洗衣机：启动→洗涤→漂洗→脱水的完整周期
    # 识别冰箱：压缩机启动和停止的周期性模式
    # 前馈网络例子：
    #
    # 区分不同电器的功率特征（微波炉
    # vs
    # 电水壶）
    # 学习复杂的非线性功率变化模式
    def forward(self, x) -> torch.Tensor:
        # x input and output shape [batch, seq_length, d_model] to meet Transformer Convention

        # 这个自注意力机制，让每个时间步都能"看到"序列中其他所有时间步的信息
        # 在NILM中：识别电器开启事件如何影响整个时间段的功率模式
        # 例如：洗衣机启动时，功率会先上升再下降，自注意力能捕获这种模式
        x = self.norm1(x)  # ← 层归一化 (在注意力前)，这个叫前置归一化，在现在比较常用
        new_x = self.attention_layer(x)  # ← 自注意力计算
        x = torch.add(x, new_x)  # ← 残差连接 (注意力输出 + 原始输入)

        # 这个是前馈网络，对自注意力输出的每个位置进行深度特征提取
        # 在NILM中：进一步处理功率特征，识别复杂的电器用电模式
        # 例如：区分微波炉的短时高功率和冰箱的周期性低功率
        x = self.norm2(x)  # ← 层归一化 (在前馈网络前)
        new_x = self.pffn(x)  # ← 前馈网络计算
        # 加上残差的大概作用是既有基本功耗特征，又能有时序模式信息
        x = torch.add(x, self.dropout(new_x))  # ← 残差连接 (前馈网络输出 + 注意力输出)

        return x
