import torch
import torch.nn as nn

from Models.NILMFormers.Layers.Transformer import EncoderLayer

class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, stride=4, in_chans=1, embed_dim=64):
        super().__init__()
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class NILMFormerAbEmbedPatch(nn.Module):
    def __init__(self,
                 window_size=128, patch_size=8, stride=4,
                 c_in=1, c_out=1,
                 instance_norm=True, learn_stats=True,
                 n_encoder_layers=3,
                 d_model=128, dp_rate=0.2,
                 att_param={'att_mask_diag': True, 'att_mask_flag': False, 'att_learn_scale': False,
                            'activation': 'gelu', 'pffn_ratio': 4, 'n_head': 8, 
                            'prenorm': True, 'norm': "LayerNorm", 
                            'store_att': False, 'dp_rate': 0.2, 'attn_dp_rate': 0.2}):
        """
        NILMFormer model use for the ablation study conducted in Section 5.4.3 : Embedding Block

        For simple Conv and Linear embedding, refer to the "NILMFormerAbEmbed" version.
        """
        super().__init__()
  
        self.d_model       = d_model
        self.c_in          = c_in
        self.c_out         = c_out
        self.window_size   = window_size
        self.patch_size    = patch_size

        self.instance_norm = instance_norm
        self.learn_stats   = learn_stats
        
        #============ Patch Embedding ============#
        self.PatchEmbed = PatchEmbed(seq_len=window_size, patch_size=patch_size, stride=stride, in_chans=c_in, embed_dim=d_model)
            
        #============ Positional Encoding ============#
        # As in PatchTST, we use a Fully learnable PE
        self.PosEncoding = nn.Parameter(torch.zeros(1, self.PatchEmbed.num_patches, d_model), requires_grad=True)
        self.PE_drop     = nn.Dropout(p=dp_rate)

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
        self.DownstreamTaskHead  = nn.Sequential(nn.Flatten(start_dim=1),
                                                 nn.Linear(in_features=self.PatchEmbed.num_patches * d_model, out_features=window_size),
                                                 nn.Dropout(0.2))

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
        encoding = x[:, 1:, :] # B 1 window_size
        x        = x[:, :1, :] # B e window_size

        # === Instance Normalization === #
        if self.instance_norm:
            inst_mean = torch.mean(x, dim=-1, keepdim=True).detach() # Mean: B 1 1
            inst_std  = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6).detach() # STD: B 1 1

            x = (x - inst_mean) / inst_std # Instance z-norm: B 1 1

        x = self.PatchEmbed(x) # B n_patch L
        x = self.PE_drop(x + self.PosEncoding) # B n_patch L
        
    
        # === Mean and Std projection === #
        stats_token = self.ProjStats1(torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)) # B 1 D
        x = torch.cat([x, stats_token], dim=1) # Add stats token B L+1 D

        # === Forward Transformer Encoder === #
        x = self.EncoderBlock(x)
        x = x[:, :-1, :] # Remove stats token: B L D

        # === Conv Head === #
        x = self.DownstreamTaskHead(x).unsqueeze(1) # B 1 window_size

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