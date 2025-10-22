#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - NILMFormer Model Architecture
#
#################################################################################################################

import torch
import torch.nn as nn

from src.nilmformer.layers.transformer import EncoderLayer
from src.nilmformer.layers.embedding import DilatedBlock

from src.nilmformer.congif import NILMFormerConfig

# Input → Instance Norm → Dilated Conv Embedding → Time Feature Projection →
# Statistical Tokens → Diagonal-Masked Multi-Head Attention → Feed Forward →
# Conv Head → Reverse Instance Norm → Output
# 输入序列 → [改进的Input Embedding] → [Transformer Encoder × N] → [卷积输出头] → 输出序列
class NILMFormer(nn.Module):
    def __init__(self, NFConfig: NILMFormerConfig):
        super().__init__()

        # ======== Validate some constraints ========#
        assert NFConfig.d_model % 4 == 0, "d_model must be divisible by 4."

        self.NFConfig = NFConfig

        c_in = NFConfig.c_in
        c_embedding = NFConfig.c_embedding
        c_out = NFConfig.c_out
        kernel_size = NFConfig.kernel_size
        kernel_size_head = NFConfig.kernel_size_head
        dilations = NFConfig.dilations
        conv_bias = NFConfig.conv_bias
        n_encoder_layers = NFConfig.n_encoder_layers
        d_model = NFConfig.d_model

        # ============ Embedding（输入嵌入层）============#
        # NILMFormer 的 Input Embedding 由三个组件组成：

        # 一共96维，拆了24维给时间相关信息了
        d_model_ = 3 * d_model // 4  # e.g., if d_model=96 => d_model_=72

        # 【Embedding 组件 1】：DilatedBlock - 处理负荷曲线（主信号）
        # 作用：用扩张卷积提取负荷曲线的多尺度时序特征
        # 输入：原始负荷曲线 (B, 1, L) - 这是电器的总功率消耗时间序列
        # 输出：嵌入特征 (B, 72, L) 当 d_model=96
        # 关键：通过不同扩张率 [1,2,4,8] 捕获从局部到长期的时序模式
        ## 总功耗序列
        self.EmbedBlock = DilatedBlock(
            c_in=c_in,
            c_out=d_model_,
            kernel_size=kernel_size,
            dilation_list=dilations,
            bias=conv_bias,
        )

        # 【Embedding 组件 2】：ProjEmbedding - 处理时间特征（辅助信号）
        # 作用：投影外生变量（时间编码：minute, hour, dow, month）
        # 输入：外生特征 (B, c_embedding, L) - 时间上下文信息
        #       例如：凌晨3点 vs 晚上8点，工作日 vs 周末
        # 输出：投影特征 (B, 24, L) 当 d_model=96
        # 关键：帮助模型理解"什么时候"容易使用某些电器
        ## 时间特征编码，提供时间上下文
        self.ProjEmbedding = nn.Conv1d(
            in_channels=c_embedding,
            out_channels=d_model // 4,
            kernel_size=1
        )

        # 【Embedding 组件 3】：ProjStats - 处理统计信息（全局上下文）
        # 用于 Instance Normalization 的可学习投影
        self.ProjStats1 = nn.Linear(2, d_model)  # (mean, std) -> d_model
        self.ProjStats2 = nn.Linear(d_model, 2)  # 反向投影用于去归一化

        # ============ Encoder ============#
        layers = []
        for _ in range(n_encoder_layers):
            layers.append(EncoderLayer(NFConfig))
        layers.append(nn.LayerNorm(d_model))
        self.EncoderBlock = nn.Sequential(*layers)

        # ============ Downstream Task Head ============#
        self.DownstreamTaskHead = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=kernel_size_head,
            padding=kernel_size_head // 2,
            padding_mode="replicate",
        )

        # ============ Initialize Weights ============#
        self.initialize_weights()

# region 初始化权重
    def initialize_weights(self):
        """Initialize nn.Linear and nn.LayerNorm weights."""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight) # 线性层使用Xavier均匀初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # 偏置初始化为0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0) # 层归一化偏置初始化为0
            nn.init.constant_(m.weight, 1.0) # 层归一化权重初始化为1

# endregion


    # 控制训练时参数更新和计算梯度 冻结或解冻模型参数
    # 迁移学习、分阶段训练或微调特定组件等高级训练策略
    def freeze_params(self, model_part, rq_grad=False):
        """Utility to freeze/unfreeze parameters in a given model part."""
        for _, child in model_part.named_children():
            for param in child.parameters():
                param.requires_grad = rq_grad # 控制梯度计算
            self.freeze_params(child) # 递归应用到子模块

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass for NILMFormer.
        Input shape: (B, 1 + e, L)
          - B: batch size
          - 1: channel for load curve
          - e: # exogenous input channels
          - L: sequence length
        """
        # Separate the channels:
        #   x[:, :1, :] => load curve
        #   x[:, 1:, :] => exogenous input(s)
        # 分离负荷曲线和时间特征
        encoding = x[:, 1:, :]  # shape: (B, e, L) # 时间特征
        x = x[:, :1, :]  # shape: (B, 1, L) # 负荷曲线

        # === Instance Normalization === #
        inst_mean = torch.mean(x, dim=-1, keepdim=True).detach()
        inst_std = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6
        ).detach()

        x = (x - inst_mean) / inst_std  # shape still (B, 1, L)

        # === Embedding === #
        # 1) 扩张卷积嵌入负荷特征
        # 1) Dilated Conv block
        x = self.EmbedBlock(
            x
        )  # shape: (B, [d_model_], L) => typically (B, 72, L) if d_model=96
        # 2) Project exogenous features
        # 2) 投影时间特征
        encoding = self.ProjEmbedding(encoding)  # shape: (B, d_model//4, L)
        # 3) Concatenate
        # 将负荷特征x和时间特征encoding拼接
        x = torch.cat([x, encoding], dim=1).permute(0, 2, 1)  # (B, L, d_model)

        # === Mean/Std tokens === #
        # 添加均值和标准差的特殊令牌
        stats_token = self.ProjStats1(
            torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)
        )  # (B, 1, d_model)
        x = torch.cat([x, stats_token], dim=1)  # (B, L + 1, d_model)

        # === Transformer Encoder === #
        x = self.EncoderBlock(x)  # (B, L + 1, d_model)
        x = x[:, :-1, :]  # remove stats token => (B, L, d_model)

        # === Conv Head === #
        x = x.permute(0, 2, 1)  # (B, d_model, L)
        x = self.DownstreamTaskHead(x)  # (B, c_out, L)

        # === Reverse Instance Normalization === #
        # stats_out => shape (B, 1, 2)
        stats_out = self.ProjStats2(stats_token)  # stats_token was (B, 1, d_model)
        outinst_mean = stats_out[:, :, 0].unsqueeze(-1)  # (B, 1, 1)
        outinst_std = stats_out[:, :, 1].unsqueeze(-1)  # (B, 1, 1)

        x = x * outinst_std + outinst_mean
        return x
