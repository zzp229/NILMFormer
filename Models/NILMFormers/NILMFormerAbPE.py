import numpy as np
import math

import torch
import torch.nn as nn

from Models.NILMFormers.Layers.Transformer import EncoderLayer
from Models.NILMFormers.Layers.ConvLayer import DilatedBlock


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return tensor + self.cached_penc


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels
    
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, dp_rate=0.2):
        super(LearnablePositionalEncoding, self).__init__()
        W_pos = torch.empty((seq_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, x):
        return self.dropout(x + self.W_pos)
    
    
class LearnablePositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding1D, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return x + self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class tAPE(nn.Module):
    """
    Improving Position Encoding of Transformers for Multivariate Time Series Classification (ConvTran)
    
    Positional Encodign scaled by window length.
    
    Code from https://github.com/Navidfoumani/ConvTran
    """
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
    def forward(self, x):
        return self.dropout(x + self.pe)

# ======================= NILMFormer =======================#
class NILMFormerAbPE(nn.Module):
    def __init__(self, 
                 type_pe='noencoding', window_size=128,
                 c_in=1, c_out=1, n_encoder_layers=3, d_model=128, 
                 instance_norm=True, learn_stats=True,
                 kernel_size=3, kernel_size_head=3, dilations=[1, 2, 4, 8], conv_bias=True,
                 att_param={'att_mask_diag': True, 'att_mask_flag': False, 'att_learn_scale': False,
                            'activation': 'gelu', 'pffn_ratio': 4, 'n_head': 8, 
                            'prenorm': True, 'norm': "LayerNorm", 
                            'store_att': False, 'dp_rate': 0.2, 'attn_dp_rate': 0.2}):
        """
        NILMFormer model use for the ablation study conducted in Section 5.4.2 : Positional Encoding

        Compare different PE used for Time Series Transformer: no encoding, fixed, tAPE, or fully learnable
        """
        super().__init__()

        self.d_model       = d_model
        self.c_in          = c_in
        self.c_out         = c_out

        self.instance_norm = instance_norm
        self.learn_stats   = learn_stats
                
        #============ Embedding ============#
        self.EmbedBlock = DilatedBlock(c_in=c_in, c_out=d_model, kernel_size=kernel_size, dilation_list=dilations, bias=conv_bias)

        #============ Positional Encoding Encoder ============#
        if type_pe == 'learnable':
            self.PosEncoding = LearnablePositionalEncoding1D(d_model, max_len=window_size)
        elif type_pe == 'fixed':
            self.PosEncoding = PositionalEncoding1D(d_model)
        elif type_pe == 'tAPE':
            self.PosEncoding = tAPE(d_model, max_len=window_size)
        elif type_pe == 'noencoding':
            self.PosEncoding = None
        else:
            raise ValueError('Type of encoding {} unknown, only "learnable", "fixed" or "noencoding" supported.'
                             .format(type_pe))
        
        if self.instance_norm:
            self.ProjStats1 = nn.Linear(2, d_model)
            self.ProjStats2 = nn.Linear(d_model, 2)
        
        #============ Encoder ============#
        layers = []
        for _ in range(n_encoder_layers):
            layers.append(EncoderLayer(d_model, 
                                       d_ff=d_model * att_param['pffn_ratio'], n_heads=att_param['n_head'], 
                                       dp_rate=att_param['dp_rate'], attn_dp_rate=att_param['attn_dp_rate'], 
                                       att_mask_diag=att_param['att_mask_diag'], 
                                       att_mask_flag=att_param['att_mask_flag'], 
                                       learnable_scale=att_param['att_learn_scale'], 
                                       store_att=att_param['store_att'],  
                                       norm=att_param['norm'], prenorm=att_param['prenorm'], 
                                       activation=att_param['activation']))
        layers.append(nn.LayerNorm(d_model))
        self.EncoderBlock = torch.nn.Sequential(*layers)
          
        #============ Downstream Task Head ============#
        self.DownstreamTaskHead = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=kernel_size_head, padding=kernel_size_head//2, padding_mode='replicate')

        #============ Initializing weights ============#        
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def freeze_params(self, model_part, rq_grad=False):
        for _, child in model_part.named_children():
            for param in child.parameters():
                param.requires_grad = rq_grad
            self.freeze_params(child)
    
    def forward(self, x) -> torch.Tensor:
        # Input as B 1+e L 
        # Separate load curve and embedding
        #encoding     = x[:, 1:, :] # B 1 L
        x  = x[:, :1, :] # B e L

        # === Instance Normalization === #
        if self.instance_norm:
            inst_mean = torch.mean(x, dim=-1, keepdim=True).detach() # Mean: B 1 1
            inst_std  = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6).detach() # STD: B 1 1

            x = (x - inst_mean) / inst_std # Instance z-norm: B 1 1
        
        # === Embedding === # 
        # Conv Dilated Embedding Block for aggregate
        x = self.EmbedBlock(x).permute(0, 2, 1) # B L D
        # Conv1x1 for encoded time features
        if self.PosEncoding is not None:
            x = self.PosEncoding(x)

        # === Mean and Std projection === #
        if self.instance_norm:
            stats_token = self.ProjStats1(torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)) # B 1 D
            x = torch.cat([x, stats_token], dim=1) # Add stats token B L+1 D

        # === Forward Transformer Encoder === #
        x = self.EncoderBlock(x) 
        if self.instance_norm:
            x = x[:, :-1, :] # Remove stats token: B L D

        # === Conv Head === #
        x = x.permute(0, 2, 1) # B D L
        x = self.DownstreamTaskHead(x) # B 1 L

        # === Reverse Instance Normalization === #
        if self.instance_norm:
            if self.learn_stats:
                # Proj back stats_token stats to get mean' and std' if learnable stats
                stats_out    = self.ProjStats2(stats_token) # B 1 2
                outinst_mean = stats_out[:, :, 0].unsqueeze(1) # B 1 1
                outinst_std  = stats_out[:, :, 1].unsqueeze(1) # B 1 1

                x = x * outinst_mean + outinst_std
            else:
                x = x * inst_mean + inst_std

        return x