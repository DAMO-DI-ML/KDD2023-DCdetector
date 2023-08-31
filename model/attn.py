import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os
from einops import rearrange, reduce, repeat


class DAC_structure(nn.Module):
    def __init__(self, win_size, patch_size, channel, mask_flag=True, scale=None, attention_dropout=0.05, output_attention=False):
        super(DAC_structure, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def forward(self, queries_patch_size, queries_patch_num, keys_patch_size, keys_patch_num, values, patch_index, attn_mask):
                                                 
        # Patch-wise Representation
        B, L, H, E = queries_patch_size.shape #batch_size*channel, patch_num, n_head, d_model/n_head
        scale_patch_size = self.scale or 1. / sqrt(E)
        scores_patch_size = torch.einsum("blhe,bshe->bhls", queries_patch_size, keys_patch_size) #batch*ch, nheads, p_num, p_num   
        attn_patch_size = scale_patch_size * scores_patch_size
        series_patch_size = self.dropout(torch.softmax(attn_patch_size, dim=-1)) # B*D_model H N N
        
        # In-patch Representation
        B, L, H, E = queries_patch_num.shape #batch_size*channel, patch_size, n_head, d_model/n_head
        scale_patch_num = self.scale or 1. / sqrt(E)
        scores_patch_num = torch.einsum("blhe,bshe->bhls", queries_patch_num, keys_patch_num) #batch*ch, nheads, p_size, p_size 
        attn_patch_num = scale_patch_num * scores_patch_num
        series_patch_num = self.dropout(torch.softmax(attn_patch_num, dim=-1)) # B*D_model H S S 

        # Upsampling
        series_patch_size = repeat(series_patch_size, 'b l m n -> b l (m repeat_m) (n repeat_n)', repeat_m=self.patch_size[patch_index], repeat_n=self.patch_size[patch_index])    
        series_patch_num = series_patch_num.repeat(1,1,self.window_size//self.patch_size[patch_index],self.window_size//self.patch_size[patch_index]) 
        series_patch_size = reduce(series_patch_size, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)
        series_patch_num = reduce(series_patch_num, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)


        if self.output_attention:
            return series_patch_size, series_patch_num
        else:
            return (None)
                                                  


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.window_size = win_size
        self.n_heads = n_heads 
        
        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)      
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask):
        
        # patch_size
        B, L, M = x_patch_size.shape
        H = self.n_heads
        queries_patch_size, keys_patch_size = x_patch_size, x_patch_size
        queries_patch_size = self.patch_query_projection(queries_patch_size).view(B, L, H, -1) 
        keys_patch_size = self.patch_key_projection(keys_patch_size).view(B, L, H, -1) 

        # patch_num
        B, L, M = x_patch_num.shape
        queries_patch_num, keys_patch_num = x_patch_num, x_patch_num
        queries_patch_num = self.patch_query_projection(queries_patch_num).view(B, L, H, -1) 
        keys_patch_num = self.patch_key_projection(keys_patch_num).view(B, L, H, -1)
        
        # x_ori
        B, L, _ = x_ori.shape
        values = self.value_projection(x_ori).view(B, L, H, -1)
        
        series, prior = self.inner_attention(
            queries_patch_size, queries_patch_num,
            keys_patch_size, keys_patch_num,
            values, patch_index,
            attn_mask
        )
        
        return series, prior
