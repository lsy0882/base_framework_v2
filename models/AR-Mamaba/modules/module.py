import sys
sys.path.append('../')

import torch
import warnings
warnings.filterwarnings('ignore')

from utils.decorators import *
from .network import *


class AudioEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int, bias: bool):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups, bias=bias)
        self.gelu = torch.nn.GELU()
    
    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, dim=0) if len(x.shape) == 1 else torch.unsqueeze(x, dim=1) # [T] - >[1, T] OR [B, T] -> [B, 1, T]
        x = self.conv1d(x)
        x = self.gelu(x)
        return x
    
class FeatureProjector(torch.nn.Module):
    def __init__(self, num_channels: int, in_channels: int, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-8)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
    
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.conv1d(x)
        return x


class Separator(torch.nn.Module):
    def __init__(self, num_stages: int, relative_positional_encoding: dict, enc_stage: dict, spk_split_stage: dict, dec_stage: dict):
        super().__init__()
        
        class RelativePositionalEncoding(torch.nn.Module):
            def __init__(self, in_channels: int, num_heads: int, maxlen: int, embed_v=False):
                super().__init__()
                self.in_channels = in_channels
                self.num_heads = num_heads
                self.embedding_dim = self.in_channels // self.num_heads
                self.maxlen = maxlen
                self.pe_k = torch.nn.Embedding(num_embeddings=2*maxlen, embedding_dim=self.embedding_dim)
                self.pe_v = torch.nn.Embedding(num_embeddings=2*maxlen, embedding_dim=self.embedding_dim) if embed_v else None
            
            def forward(self, pos_seq: torch.Tensor):
                pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
                pos_seq += self.maxlen
                pe_k_output = self.pe_k(pos_seq)
                pe_v_output = self.pe_v(pos_seq) if self.pe_v is not None else None
                return pe_k_output, pe_v_output
        
        class SepEncStage(torch.nn.Module):
            def __init__(self, global_blocks: dict, local_blocks: dict, down_conv_layer: dict, down_conv=True):
                super().__init__()
                
                class EncGlobalBlocks(torch.nn.Module):
                    def __init__(self, num_blocks: int, in_channels: int, num_mha_heads: int, dropout_rate: float):
                        super().__init__()
                        self.blocks = torch.nn.ModuleList()
                        for _ in range(num_blocks):
                            block = torch.nn.ModuleDict({
                                'self_attn': MultiHeadAttention(
                                    n_head=num_mha_heads, n_feat=in_channels, dropout_rate=dropout_rate),
                                'linear': torch.nn.Sequential(
                                    torch.nn.LayerNorm(normalized_shape=in_channels), 
                                    torch.nn.Linear(in_features=in_channels, out_features=in_channels), 
                                    torch.nn.Sigmoid()),
                                'feed_forward': FFN(in_channels=in_channels, dropout_rate=dropout_rate)
                            })
                            self.blocks.append(block)
                    
                    def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
                        """
                        Compute encoded features.
                            :param torch.Tensor x: encoded source features (batch, max_time_in, size)
                            :param torch.Tensor mask: mask for x (batch, max_time_in)
                            :rtype: Tuple[torch.Tensor, torch.Tensor]
                        """
                        for block in self.blocks:
                            down_len = pos_k.shape[0]
                            x_down = torch.nn.functional.adaptive_avg_pool1d(input=x, output_size=down_len)
                            x = x.permute([0, 2, 1])
                            x_down = x_down.permute([0, 2, 1])
                            x_down = block['self_attn'](x_down, pos_k, None)
                            x_down = x_down.permute([0, 2, 1])
                            x_downup = torch.nn.functional.upsample(input=x_down, size=x.shape[1])
                            x_downup = x_downup.permute([0, 2, 1])
                            x = x + block['linear'](x) * x_downup
                            x = block['feed_forward'](x)
                            x = x.permute([0, 2, 1])
                        return x
                
                class EncLocalBlocks(torch.nn.Module):
                    def __init__(self, num_blocks: int, in_channels: int, num_clsa_heads: int, dropout_rate: float):
                        super().__init__()
                        self.blocks = torch.nn.ModuleList()
                        for _ in range(num_blocks):
                            block = torch.nn.ModuleDict({
                                'clsa': ConvLocalSelfAttention(in_channels, num_clsa_heads, dropout_rate),
                                'ffn': FFN(in_channels, dropout_rate)
                            })
                            self.blocks.append(block)
                    
                    def forward(self, x: torch.Tensor):
                        for block in self.blocks:
                            x = block['clsa'](x)
                            x = block['ffn'](x)
                        return x
                
                class DownConvLayer(torch.nn.Module):
                    def __init__(self, in_channels: int, samp_kernel_size: int):
                        """Construct an EncoderLayer object."""
                        super().__init__()
                        self.down_conv = torch.nn.Conv1d(
                            in_channels=in_channels, out_channels=in_channels, kernel_size=samp_kernel_size, stride=2, padding=(samp_kernel_size-1)//2, groups=in_channels)
                        self.BN = torch.nn.BatchNorm1d(num_features=in_channels)
                        self.gelu = torch.nn.GELU()
                    
                    def forward(self, x: torch.Tensor):
                        x = x.permute([0, 2, 1])
                        x = self.down_conv(x)
                        x = self.BN(x)
                        x = self.gelu(x)
                        x = x.permute([0, 2, 1])
                        return x
                
                self.g_block_1 = EncGlobalBlocks(**global_blocks)
                self.l_block_1 = EncLocalBlocks(**local_blocks)
                self.g_block_2 = EncGlobalBlocks(**global_blocks)
                self.l_block_2 = EncLocalBlocks(**local_blocks)
                self.downconv = DownConvLayer(**down_conv_layer) if down_conv == True else None
                
            def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
                '''
                x: [B, N, T]
                '''
                input = x
                x = self.g_block_1(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.g_block_2(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_2(x)
                x = x.permute(0, 2, 1).contiguous()
                skip = x
                if self.downconv:
                    x = x.permute(0, 2, 1).contiguous()
                    x = self.downconv(x)
                    x = x.permute(0, 2, 1).contiguous()
                # [BK, S, N]
                return x, skip, input
        
        class SpkSplitStage(torch.nn.Module):
            def __init__(self, in_channels: int, num_spks: int):
                super().__init__()
                self.linear = torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels, 4*in_channels*num_spks, kernel_size=1),
                    torch.nn.GLU(dim=-2),
                    torch.nn.Conv1d(2*in_channels*num_spks, in_channels*num_spks, kernel_size=1))
                self.norm = torch.nn.GroupNorm(1, in_channels, eps=1e-8)
                self.num_spks = num_spks
                
            def forward(self, x: torch.Tensor):
                x = self.linear(x)
                B, _, T = x.shape
                x = x.view(B*self.num_spks,-1, T).contiguous()
                x = self.norm(x)
                return x
        
        class SepDecStage(torch.nn.Module):
            def __init__(self, up_conv_layer: dict, global_blocks: dict, local_blocks: dict):
                super().__init__()
                
                class UpConvLayer(torch.nn.Module):
                    def __init__(self, in_channels: int, samp_kernel_size: int):
                        super().__init__()
                    
                    def forward(self, x: torch.Tensor):
                        x = x.permute([0, 2, 1])
                        x = torch.nn.functional.upsample(x, x.shape[-1]*2)
                        x = x.permute([0, 2, 1])
                        return x
                    
                class DecGlobalBlocks(torch.nn.Module):
                    def __init__(self, num_blocks, in_channels, num_mhca_heads, num_mha_heads, dropout_rate):
                        super().__init__()
                        self.blocks = torch.nn.ModuleList()
                        for _ in range(num_blocks):
                            block = torch.nn.ModuleDict({
                                'cross_attn': MultiHeadCrossAttention(
                                    n_head=num_mhca_heads, n_feat=in_channels, dropout_rate=dropout_rate),
                                'linear_sa': torch.nn.Sequential(
                                    torch.nn.LayerNorm(normalized_shape=in_channels), 
                                    torch.nn.Linear(in_features=in_channels, out_features=in_channels), 
                                    torch.nn.Sigmoid()),
                                'self_attn': MultiHeadAttention(
                                    n_head=num_mha_heads, n_feat=in_channels, dropout_rate=dropout_rate),
                                'linear_ca': torch.nn.Sequential(
                                    torch.nn.LayerNorm(normalized_shape=in_channels), 
                                    torch.nn.Linear(in_features=in_channels, out_features=in_channels), 
                                    torch.nn.Sigmoid()),
                                'ffn': FFN(in_channels=in_channels, dropout_rate=dropout_rate)})
                            self.blocks.append(block)
                    
                    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_k: torch.Tensor):
                        """
                        Compute encoded features.
                            :param torch.Tensor x: encoded source features (batch, max_time_in, size)
                            :param torch.Tensor mask: mask for x (batch, max_time_in)
                            :rtype: Tuple[torch.Tensor, torch.Tensor]
                        """
                        for block in self.blocks:
                            down_len = k.shape[-1]
                            x_down = torch.nn.functional.adaptive_avg_pool1d(x,down_len)

                            x_down = x_down.permute([0, 2, 1])
                            x = x.permute([0, 2, 1])
                            k = k.permute([0, 2, 1])
                            v = v.permute([0, 2, 1])

                            x_down = block['cross_attn'](x_down, k, v, pos_k, None)
                            
                            x_down = x_down.permute([0, 2, 1])
                            x_downup = torch.nn.functional.upsample(x_down, x.shape[1])
                            x_downup = x_downup.permute([0, 2, 1])
                            
                            x = x + block['linear_sa'](x)*x_downup

                            x = x.permute([0, 2, 1])
                            x_down = torch.nn.functional.adaptive_avg_pool1d(x,down_len)
                            x = x.permute([0, 2, 1])
                            x_down = x_down.permute([0, 2, 1])

                            x_down = block['self_attn'](x_down, pos_k, None)

                            x_down = x_down.permute([0, 2, 1])
                            x_downup = torch.nn.functional.upsample(x_down, x.shape[1])
                            x_downup = x_downup.permute([0, 2, 1])

                            x = x + block['linear_ca'](x)*x_downup

                            x = block['ffn'](x)

                            x = x.permute([0, 2, 1])
                        return x
                    
                class DecLocalBlocks(torch.nn.Module):
                    def __init__(self, num_blocks, in_channels, num_clca_heads, num_clsa_heads, dropout_rate):
                        super().__init__()
                        self.blocks = torch.nn.ModuleList()
                        for _ in range(num_blocks):
                            block = torch.nn.ModuleDict({
                                'clca': ConvLocalCrossAttention(
                                    in_channels=in_channels, num_heads=num_clca_heads, dropout_rate=dropout_rate),
                                'clsa': ConvLocalSelfAttention(
                                    in_channels=in_channels, num_heads=num_clsa_heads, dropout_rate=dropout_rate),
                                'ffn': FFN(in_channels=in_channels, dropout_rate=dropout_rate)})
                            self.blocks.append(block)

                    def forward(self, x: torch.Tensor, skip: torch.Tensor):
                        for block in self.blocks:
                            skip = block['clca'](x, skip)
                            skip = block['clsa'](skip)
                            skip = block['ffn'](skip)
                        return skip
                
                self.up_conv = UpConvLayer(**up_conv_layer)
                self.g_block_1 = DecGlobalBlocks(**global_blocks)
                self.l_block_1 = DecLocalBlocks(**local_blocks)
                self.g_block_2 = DecGlobalBlocks(**global_blocks)
                self.l_block_2 = DecLocalBlocks(**local_blocks)
            
            def forward(self, x: torch.Tensor, skip: torch.Tensor, x_bn: torch.Tensor, pos_k: torch.Tensor):
                '''
                x: [B, N, T]
                '''
                # [BS, K, H]
                x = x.permute(0, 2, 1).contiguous()
                x = self.up_conv(x)
                skip = self.g_block_1(skip, x_bn, x_bn, pos_k)
                skip = skip.permute(0, 2, 1).contiguous()
                skip = self.l_block_1(x, skip)
                skip = skip.permute(0, 2, 1).contiguous()
                skip = self.g_block_2(skip, x_bn, x_bn, pos_k)
                skip = skip.permute(0, 2, 1).contiguous()
                skip = self.l_block_2(x, skip)
                skip = skip.permute(0, 2, 1).contiguous()
                return skip
        
        self.num_stages = num_stages
        self.pos_emb = RelativePositionalEncoding(**relative_positional_encoding)
        
        # Temporal Contracting Part
        self.enc_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.enc_stages.append(SepEncStage(**enc_stage, down_conv=True))
        
        self.bottleneck_G = SepEncStage(**enc_stage, down_conv=False)
        self.spk_split_block = SpkSplitStage(**spk_split_stage)
        
        # Temporal Expanding Part
        self.dec_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.dec_stages.append(SepDecStage(**dec_stage))
    
    def forward(self, input: torch.Tensor):
        '''input: [B, N, L]'''
        # feature projection
        x, _ = self.pad_signal(input)
        len_x = x.shape[-1]
        # Temporal Contracting Part
        pos_seq = torch.arange(0, len_x//2**self.num_stages).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k, _ = self.pos_emb(pos_seq)
        skip = []
        for idx in range(self.num_stages):
            x, skip_, bn_ms_ = self.enc_stages[idx](x, pos_k)
            skip_ = self.spk_split_block(skip_)
            skip.append(skip_)
        x, _, _ = self.bottleneck_G(x, pos_k)
        x = self.spk_split_block(x) # B, 2F, T
        each_stage_outputs = []
        x_bn = x
        # Temporal Expanding Part
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = self.dec_stages[idx](x, skip[idx_en], x_bn, pos_k)
        last_stage_output = x 
        return last_stage_output, each_stage_outputs
    
    def pad_signal(self, input: torch.Tensor):
        #  (B, T) or (B, 1, T)
        if input.dim() == 1: input = input.unsqueeze(0)
        elif input.dim() not in [2, 3]: raise RuntimeError("Input can only be 2 or 3 dimensional.")
        elif input.dim() == 2: input = input.unsqueeze(1)
        L = 2**self.num_stages
        batch_size = input.size(0)  
        ndim = input.size(1)
        nframe = input.size(2)
        padded_len = (nframe//L + 1)*L
        rest = 0 if nframe%L == 0 else padded_len - nframe
        if rest > 0:
            pad = torch.autograd.Variable(torch.zeros(batch_size, ndim, rest)).type(input.type()).to(input.device)
            input = torch.cat([input, pad], dim=-1)
        return input, rest


class OutputLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_spks: int):
        super().__init__()
        # feature expansion back
        self.num_spks = num_spks
        self.end_conv1x1 = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 4*out_channels),
            torch.nn.GLU(),
            torch.nn.Linear(2*out_channels, in_channels))
            
    def forward(self, x: torch.Tensor, input: torch.Tensor):
        x = x[...,:input.shape[-1]]
        x = x.permute([0, 2, 1])
        x = self.end_conv1x1(x)
        x = x.permute([0, 2, 1])
        B, N, L = x.shape
        B = B // self.num_spks
        
        x = x.view(B, self.num_spks, N, L)
        # [spks, B, N, L]
        x = x.transpose(0, 1)
        return x


class AudioDecoder(torch.nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        # x: [B, N, L]
        if x.dim() not in [2, 3]: raise RuntimeError("{} accept 3/4D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        x = torch.squeeze(x, dim=1) if torch.squeeze(x).dim() == 1 else torch.squeeze(x)
        return x