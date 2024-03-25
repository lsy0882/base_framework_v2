import torch
import math
import numpy
from utils.decorators import *


class FFN(torch.nn.Module):
    def __init__(self, in_channels, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.LayerNorm(in_channels),
            torch.nn.Linear(in_channels, in_channels*6))
        self.depthwise = torch.nn.Conv1d(in_channels*6, in_channels*6, 3, padding=1, groups=in_channels*6)
        self.net2 = torch.nn.Sequential(
            torch.nn.GLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_channels*3, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = torch.nn.Parameter(torch.ones(1,1,in_channels)*Layer_scale_init, requires_grad=True)
        
    def forward(self, x):
        y = self.net1(x)
        y = y.permute(0, 2, 1).contiguous()
        y = self.depthwise(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.net2(y)
        return x + y*self.Layer_scale


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention layer.
        :param int n_head: the number of head s
        :param int n_feat: the number of features
        :param float dropout_rate: dropout rate
    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, Layer_scale_init=1.0e-5):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head # We assume d_v always equals d_k
        self.h = n_head
        self.layer_norm = torch.nn.LayerNorm(n_feat)
        self.linear_q = torch.nn.Linear(n_feat, n_feat)
        self.linear_k = torch.nn.Linear(n_feat, n_feat)
        self.linear_v = torch.nn.Linear(n_feat, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.Layer_scale = torch.nn.Parameter(torch.ones(1,1,n_feat)*Layer_scale_init, requires_grad=True)
    
    def forward(self, x, pos_k, mask):
        """
        Compute 'Scaled Dot Product Attention'.
            :param torch.Tensor mask: (batch, time1, time2)
            :param torch.nn.Dropout dropout:
            :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
            weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = torch.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0,1)
        if pos_k is not None:
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.dropout(self.linear_out(x))*self.Layer_scale  # (batch, time1, d_model)


class MultiHeadCrossAttention(torch.nn.Module):
    """
    Multi-Head Attention layer.
        :param int n_head: the number of head s
        :param int n_feat: the number of features
        :param float dropout_rate: dropout rate
    """
    def __init__(self, n_head, n_feat, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head # We assume d_v always equals d_k
        self.h = n_head
        self.layer_norm_q = torch.nn.LayerNorm(n_feat)
        self.layer_norm_kv = torch.nn.LayerNorm(n_feat)
        self.linear_q = torch.nn.Linear(n_feat, n_feat)
        self.linear_k = torch.nn.Linear(n_feat, n_feat)
        self.linear_v = torch.nn.Linear(n_feat, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.Layer_scale = torch.nn.Parameter(torch.ones(1,1,n_feat)*Layer_scale_init, requires_grad=True)

    def forward(self, q, k, v, pos_k, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = q.size(0)
        q = self.layer_norm_q(q)
        k = self.layer_norm_kv(k)
        v = self.layer_norm_kv(v)
        q = self.linear_q(q).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        k = self.linear_k(k).view(n_batch, -1, self.h, self.d_k)  #(b, t, d)
        v = self.linear_v(v).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = torch.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0,1)
        if pos_k is not None:
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.dropout(self.linear_out(x)) * self.Layer_scale  # (batch, time1, d_model)


class ConvLocalSelfAttention(torch.nn.Module):
    def __init__(self, in_channels, num_heads, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.num_heads = num_heads
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear1 = torch.nn.Linear(in_channels, in_channels*2)
        self.GLU = torch.nn.GLU()
        self.dim = in_channels // self.num_heads
        self.dw_conv_1d = torch.nn.ModuleList([
            torch.nn.Conv1d(self.dim, self.dim, 7, stride=1, padding='same', groups=self.dim),
            torch.nn.Conv1d(self.dim, self.dim, 15, stride=1, padding='same', groups=self.dim),
            torch.nn.Conv1d(self.dim, self.dim, 33, stride=1, padding='same', groups=self.dim),
            torch.nn.Conv1d(self.dim, self.dim, 65, stride=1, padding='same', groups=self.dim)])
        self.scale_linear = torch.nn.Conv1d(in_channels, in_channels*2, 1, groups=self.dim)
        self.BN = torch.nn.BatchNorm1d(2*self.dim*self.num_heads)
        self.GELU = torch.nn.GELU()
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(2*in_channels, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = torch.nn.Parameter(torch.ones(1,1,in_channels)*Layer_scale_init, requires_grad=True)
    
    def forward(self, x):
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.GLU(y)
        y = y.permute([0, 2, 1]) # B, F, T
        y = torch.split(y,self.dim, dim=1)
        z = []
        for i, layer in enumerate(self.dw_conv_1d):
            z.append(layer(y[i]))
        y = torch.stack(z, dim=-1) # B, F/s, T, s
        B, F, T, s = y.shape
        y = y.permute(0, 1, 3, 2).contiguous().view(B, -1, T) # F/s * s
        y = self.scale_linear(y)
        y = self.GELU(self.BN(y))
        y = y.permute(0, 2, 1) # B, T, 2F
        y = self.linear2(y)
        return x + y*self.Layer_scale


class ConvLocalCrossAttention(torch.nn.Module):
    def __init__(self, in_channels, num_heads, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.num_heads = num_heads
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.layer_norm_kv = torch.nn.LayerNorm(in_channels)
        self.linear1 = torch.nn.Linear(2*in_channels, in_channels)
        self.linear_cross = torch.nn.Linear(in_channels, in_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.dim = in_channels // self.num_heads
        self.dw_conv_1d = torch.nn.ModuleList([
            torch.nn.Conv1d(self.dim, self.dim, 7, stride=1, padding='same', groups=self.dim),
            torch.nn.Conv1d(self.dim, self.dim, 15, stride=1, padding='same', groups=self.dim),
            torch.nn.Conv1d(self.dim, self.dim, 33, stride=1, padding='same', groups=self.dim),
            torch.nn.Conv1d(self.dim, self.dim, 65, stride=1, padding='same', groups=self.dim)])
        self.scale_linear = torch.nn.Conv1d(in_channels, in_channels*2, 1, groups=self.dim)
        self.BN = torch.nn.BatchNorm1d(2*self.dim*self.num_heads)
        self.GELU = torch.nn.GELU()
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(2*in_channels, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = torch.nn.Parameter(torch.ones(1,1,in_channels)*Layer_scale_init, requires_grad=True)
        
    def forward(self, x, skip):
        x = self.layer_norm_kv(x)
        y = self.layer_norm(skip)
        y = torch.cat((x,y),dim=-1)
        y = self.linear1(y)
        x = self.linear_cross(x)
        y = self.sigmoid(y) * x
        y = y.permute([0, 2, 1])
        y = torch.split(y,self.dim, dim=1)
        z = []
        for i, layer in enumerate(self.dw_conv_1d):
            z.append(layer(y[i]))
        y = torch.stack(z, dim=-1) # B, F/s, T, s
        B, F, T, s = y.shape
        y = y.permute(0, 1, 3, 2).contiguous().view(B, -1, T) # F/s * s
        y = self.scale_linear(y)
        y = self.GELU(self.BN(y))
        y = y.permute([0, 2, 1]) # B, T, 2F
        y = self.linear2(y)
        return skip + y*self.Layer_scale