import os
import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T
from PIL import Image

import time
from tqdm import tqdm
from einops import rearrange, repeat

from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding

import numpy as np
from src.utils import *
from src.normalization import Normalization

from accelerate.utils import broadcast_object_list

# helpers functions

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class UnsqueezeLastDim(nn.Module):
    def forward(self, x):
        return torch.unsqueeze(x, -1)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim, padding_mode = 'zeros'):
    if padding_mode == 'zeros':
        return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1), padding_mode='zeros')
    elif padding_mode == 'circular':
        return CircularUpsample(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
    elif padding_mode == 'circular_1d':
        return Circular_1d_Upsample(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

# WARNING: (Experimental) This is hard-coded for above kernel size, stride, and padding. Do not use for other cases.
# Use this for upsamling with circular padding in both pixel dimensions (Torch does not offer this natively).
class CircularUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(CircularUpsample, self).__init__()
        assert kernel_size[0] == 1 and kernel_size[1] == 4 and kernel_size[2] == 4
        assert stride[0] == 1 and stride[1] == 2 and stride[2] == 2
        assert padding[0] == 0 and padding[1] == 1 and padding[2] == 1
        assert dilation == 1
        if not isinstance(dilation, tuple):
            dilation = (dilation, dilation, dilation)
        self.true_padding = (dilation[0] * (kernel_size[0] - 1) - padding[0],
                             dilation[1] * (kernel_size[1] - 1) - padding[1],
                             dilation[2] * (kernel_size[2] - 1) - padding[2])
        # this ensures that no padding is applied by the ConvTranspose3d layer since we manually apply it before
        self.removed_padding = (dilation[0] * (kernel_size[0] - 1) + stride[0] + padding[0] - 1,
                             dilation[1] * (kernel_size[1] - 1) + stride[1] + padding[1] - 1,
                             dilation[2] * (kernel_size[2] - 1) + stride[2] + padding[2] - 1)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding=self.removed_padding)

    def forward(self, x):
        true_padding_repeated = tuple(i for i in reversed(self.true_padding) for _ in range(2))
        x = nn.functional.pad(x, true_padding_repeated, mode = 'circular') # manually apply padding of 1 on all sides
        x = self.conv_transpose(x)
        return x

# WARNING: (Experimental) This is hard-coded for above kernel size, stride, and padding. Do not use for other cases.
# Use this for upsamling with circular padding in horizontal pixel dimension (Torch does not offer this natively).
class Circular_1d_Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(Circular_1d_Upsample, self).__init__()
        assert kernel_size[0] == 1 and kernel_size[1] == 4 and kernel_size[2] == 4
        assert stride[0] == 1 and stride[1] == 2 and stride[2] == 2
        assert padding[0] == 0 and padding[1] == 1 and padding[2] == 1
        assert dilation == 1
        if not isinstance(dilation, tuple):
            dilation = (dilation, dilation, dilation)
        self.true_padding = (dilation[0] * (kernel_size[0] - 1) - padding[0],
                             dilation[1] * (kernel_size[1] - 1) - padding[1],
                             dilation[2] * (kernel_size[2] - 1) - padding[2])
        # this ensures that no padding is applied by the ConvTranspose3d layer since we manually apply it before
        self.removed_padding = (dilation[0] * (kernel_size[0] - 1) + stride[0] + padding[0] - 1,
                             dilation[1] * (kernel_size[1] - 1) + stride[1] + padding[1] - 1,
                             dilation[2] * (kernel_size[2] - 1) + stride[2] + padding[2] - 1)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding=self.removed_padding)

    def forward(self, x):
        true_padding_repeated = tuple(i for i in reversed(self.true_padding) for _ in range(2))
        # NOTE dim=-1 is horizontal (PBC), dim=-2 is vertical, F.pad starts from last dim, so we take circular padding as first entry and zero padding as second entry      
        true_padding_repeated_horizontal = true_padding_repeated[0:2] + (0,) * (len(true_padding_repeated) - 2)
        true_padding_repeated_vertical = (0,) * 2 + true_padding_repeated[2:]
        x = nn.functional.pad(x, true_padding_repeated_horizontal, mode = 'circular') # manually apply padding of 1 on all sides
        x = nn.functional.pad(x, true_padding_repeated_vertical, mode = 'constant') # manually apply padding of 1 on all sides
        x = self.conv_transpose(x)
        return x

# (Experimental) Use this for downsampling with circular padding in horizontal pixel dimension (Torch does not offer this natively).
class Circular_1d_Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """Initialize the class with explicit zero-padding for 3D convolution."""
        super(Circular_1d_Conv3d, self).__init__()
        self.true_padding = padding
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0)

    def forward(self, x):
        """Applies 3D convolution with explicit zero-padding."""
        # Calculate the number of padding depth, rows and columns 
        # NOTE dim=-1 is horizontal (PBC), dim=-2 is vertical, F.pad starts from last dim, so we take circular padding as first entry and zero padding as second entry
        true_padding_repeated = tuple(i for i in reversed(self.true_padding) for _ in range(2))
        # NOTE dim=-1 is horizontal (PBC), dim=-2 is vertical, F.pad starts from last dim, so we take circular padding as first entry and zero padding as second entry
        true_padding_repeated_horizontal = true_padding_repeated[0:2] + (0,) * (len(true_padding_repeated) - 2)
        true_padding_repeated_vertical = (0,) * 2 + true_padding_repeated[2:]
        x = nn.functional.pad(x, true_padding_repeated_horizontal, mode = 'circular') # manually apply padding of 1 on all sides
        x = nn.functional.pad(x, true_padding_repeated_vertical, mode = 'constant') # manually apply padding of 1 on all sides
        # Apply 3D convolution
        x = self.conv(x)        
        return x

def Downsample(dim, padding_mode='zeros'):
    if padding_mode == 'zeros' or padding_mode == 'circular':
        return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1), padding_mode=padding_mode)
    elif padding_mode == 'circular_1d':
        return Circular_1d_Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, padding_mode = 'zeros', groups = 8):
        super().__init__()
        if padding_mode == 'zeros' or padding_mode == 'circular':
            self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1), padding_mode=padding_mode)
        elif padding_mode == 'circular_1d':
            self.proj = Circular_1d_Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, padding_mode = 'zeros', groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, padding_mode = padding_mode, groups = groups)
        self.block2 = Block(dim_out, dim_out, padding_mode = padding_mode, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, cond_attention = None, cond_dim = 64, per_frame_cond = False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias = False)
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias=False)
        
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

        self.cond_attention = cond_attention

        self.per_frame_cond = per_frame_cond

    def forward(self, x, label_emb_mm = None):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        if self.cond_attention == 'none' or label_emb_mm == None:
            qkv = self.to_qkv(x).chunk(3, dim = 1)
            q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)      
        elif self.cond_attention == 'self-stacked':
            qkv = self.to_qkv(x).chunk(3, dim = 1)
            q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
            ek = self.to_k(label_emb_mm) # this acts on last dim
            ev = self.to_v(label_emb_mm) # this acts on last dim
            if self.per_frame_cond:  # we do per-frame-conditioning, otherwise we condition on the whole video with positional bias
                # in spatial attention, we get in b, 11, 1, c, i.e., we align the 11 frames of x [b, b2, n, c] - 'b2' with the 11 frames of label_emb_mm [batch x frames x embedding] - 'frames'
                # add single token (n=1) add correct dimension
                ek, ev = map(lambda t: repeat(t, 'b f x -> b f 1 x'), (ek, ev))
            else:
                # repeat ek and ev along frames/pixels for agnostic attention, also holds for per-frame-conditioning in temporal attention (f=n), where we broadcast time signal to all pixels
                ek, ev = map(lambda t: repeat(t, 'b n x -> b f n x', f = f), (ek, ev))
            # rearrange so that linear layer without bias corresponds to head and head_dim
            ek, ev = map(lambda t: rearrange(t, 'b f n (h c) -> (b f) h c n', h = self.heads), (ek, ev))
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
        elif self.cond_attention == 'cross-attention':
            q = self.to_q(x)
            q = rearrange(q, 'b (h c) x y -> b h c (x y)', h = self.heads)
            k = self.to_k(label_emb_mm) # this acts on last dim
            v = self.to_v(label_emb_mm) # this acts on last dim
            # rearrange so that linear layer without bias corresponds to head and head_dim
            # repeat ek and ev along frame dimension (x is (h w)
            k, v = map(lambda t: repeat(t, 'b n x -> b f n x', f = f), (k, v))
            # treat frames as batches and split x into heads and head_dim
            k, v = map(lambda t: rearrange(t, 'b f n (h c) -> (b f) h c n', h = self.heads), (k, v))
        else:
            raise ValueError('cond_attention must be none, self-stacked or cross-attention')

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w) # added this (not included in original repo)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None, 
        cond_attention = None, 
        cond_dim = 64,
        per_frame_cond = False,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.cond_attention = cond_attention # none, stacked self-attention or cross-attention
        
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

        self.per_frame_cond = per_frame_cond

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None, # we do not care about this
        label_emb_mm = None
    ):
        # b is batch, b2 is either (h w) or f which will be treated as batch, n is the token, c the dim from which we build the heads and dim_head
        b, b2, n, c = x.shape
        device = x.device

        if self.cond_attention == 'none' or label_emb_mm == None:
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            if exists(focus_present_mask) and focus_present_mask.all():
                # if all batch samples are focusing on present
                # it would be equivalent to passing that token's values through to the output
                values = qkv[-1]
                return self.to_out(values)

            # split out heads
            # n are the input tokens, which can be pixels or frames 
            # depending on whether we are in spatial or temporal attention
            q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)
            if exists(self.rotary_emb):
                k = self.rotary_emb.rotate_queries_or_keys(k)

        elif self.cond_attention == 'self-stacked':

            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)
            if exists(self.rotary_emb):
                k = self.rotary_emb.rotate_queries_or_keys(k)
            ek = self.to_k(label_emb_mm) # this acts on last dim
            ev = self.to_v(label_emb_mm) # this acts on last dim
            if pos_bias is None and self.per_frame_cond:  # indicator of spatial attention -> we do per-frame-conditioning, otherwise we condition on the whole video with positional bias
                # in spatial attention, we get in b, 11, 1, c, i.e., we align the 11 frames of x [b, b2, n, c] - 'b2' with the 11 frames of label_emb_mm [batch x frames x embedding] - 'frames'
                # add single token (n=1) add correct dimension
                ek, ev = map(lambda t: repeat(t, 'b f c -> b f 1 c'), (ek, ev))
            else:
                # repeat ek and ev along frames/pixels for agnostic attention, also holds for per-frame-conditioning in temporal attention (f=n), where we broadcast time signal to all pixels
                ek, ev = map(lambda t: repeat(t, 'b n c -> b b2 n c', b2 = b2), (ek, ev))
            # rearrange so that linear layer without bias corresponds to head and head_dim
            ek, ev = map(lambda t: rearrange(t, 'b b2 n (h d) -> b b2 h n d', h = self.heads), (ek, ev))
            
            # add rotary embedding to ek if we have temporal attention and per-frame-conditioning since we want to encode the temporal information in the conditioning
            if exists(self.rotary_emb) and self.per_frame_cond:
                ek = self.rotary_emb.rotate_queries_or_keys(ek)

            k = torch.cat([ek, k], dim=-2)
            v = torch.cat([ev, v], dim=-2)

        elif self.cond_attention == 'cross-attention':
            q = self.to_q(x)
            q = rearrange(q, '... n (h d) -> ... h n d', h = self.heads)
            k = self.to_k(label_emb_mm) # this acts on last dim
            v = self.to_v(label_emb_mm) # this acts on last dim
            # rearrange so that linear layer without bias corresponds to head and head_dim
            # repeat ek and ev along frame dimension (x is (h w)
            k, v = map(lambda t: repeat(t, 'b n c -> b b2 n c', b2 = b2), (k, v))
            # treat frames as batches and split x into heads and head_dim
            k, v = map(lambda t: rearrange(t, 'b b2 n (h d) -> b b2 h n d', h = self.heads), (k, v))

        else:
            raise ValueError('cond_attention must be none, self-stacked or cross-attention')

        # scale
        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)

        # similarity
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias
        if exists(pos_bias):
            if self.cond_attention == 'self-stacked' and exists(label_emb_mm):
                # add relative positional bias to similarity for last n entries of last dimension (those that relate to frames and not cond_emb)
                sim[:, :, :, :, -n:] = sim[:, :, :, :, -n:] + pos_bias
                if self.per_frame_cond:
                    # add positional bias to conditioning since this is inside a temporal attention if (due to pos_bias)
                    # add relative positional bias two similarity for first n entries of last dimension (those that relate to cond_emb)
                    # this assumes that we can bias the (per-frame-) conditioning in the same way as the frames 
                    sim[:, :, :, :, :n] = sim[:, :, :, :, :n] + pos_bias
            else:
                sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# convolutional encoder for 1D stress-strain response (only required for ablation study)
class SignalEmbedding(nn.Module):
    def __init__(self, cond_arch, init_channel, channel_upsamplings):
        super().__init__()
        if cond_arch == 'CNN':
            scale_factor = [init_channel, *map(lambda m: 1 * m, channel_upsamplings)]
            in_out_channels = list(zip(scale_factor[:-1], scale_factor[1:]))
            self.num_resolutions = len(in_out_channels)
            self.emb_model = self.generate_conv_embedding(in_out_channels)
        elif cond_arch == 'GRU':
            self.emb_model = nn.GRU(input_size = init_channel, hidden_size = channel_upsamplings[-1], num_layers = 3, batch_first=True)
        else:
            raise ValueError('Unknown architecture: {}'.format(cond_arch))

        self.cond_arch = cond_arch

    def Downsample1D(self, dim, dim_out = None):
        return nn.Conv1d(dim,default(dim_out, dim),kernel_size=4, stride=2, padding=1)

    def generate_conv_embedding(self, channel_upsamplings):
        embedding_modules = nn.ModuleList([])
        for idx, (ch_in, ch_out) in enumerate(channel_upsamplings):
            embedding_modules.append(self.Downsample1D(ch_in,ch_out))
            embedding_modules.append(nn.SiLU())
        return nn.Sequential(*embedding_modules)

    def forward(self, x):
        # add channel dimension for conv1d
        if len(x.shape) == 2 and self.cond_arch == 'CNN':
            x = x.unsqueeze(1)
            x = self.emb_model(x)
        elif len(x.shape) == 2 and self.cond_arch == 'GRU':
            x = x.unsqueeze(2)
            x, _ = self.emb_model(x)
        x = torch.squeeze(x)
        return x

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        attn_heads = 8,
        attn_dim_head = 32,
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
        resnet_groups = 8,
        cond_bias = False,
        cond_attention = 'none', # 'none', 'self-stacked', 'cross', 'self-cross/spatial'
        cond_attention_tokens = 6,
        cond_att_GRU = False,
        use_temporal_attention_cond = False,
        cond_to_time = 'add',
        per_frame_cond = False,
        padding_mode = 'zeros',
    ):
        super().__init__()
        self.channels = channels

        time_dim = dim * 4

        self.cond_bias = cond_bias
        self.cond_attention = cond_attention if not per_frame_cond else 'self-stacked'
        self.cond_attention_tokens = cond_attention_tokens if not per_frame_cond else 11
        self.cond_att_GRU = cond_att_GRU
        self.cond_dim = time_dim
        self.use_temporal_attention_cond = use_temporal_attention_cond
        self.cond_to_time = cond_to_time
        self.per_frame_cond = per_frame_cond
        self.padding_mode = padding_mode

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        # this reshapes a tensor of shape [first argument] to
        # [second argument], applies an attention layer and then transforms it back 
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb, cond_attention = self.cond_attention, cond_dim = self.cond_dim, per_frame_cond = per_frame_cond))

        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32)

        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2

        if self.padding_mode == 'zeros' or self.padding_mode == 'circular':
            self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding), padding_mode=self.padding_mode)
        elif self.padding_mode == 'circular_1d':
            self.init_conv = Circular_1d_Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # CNN signal embedding for cond bias
        self.sign_emb_CNN = SignalEmbedding('CNN', init_channel=1, channel_upsamplings=(16, 32, 64, 128, self.cond_dim))
        # GRU signal embedding
        self.sign_emb_GRU = None
        if cond_att_GRU:
            self.sign_emb_GRU = SignalEmbedding('GRU', init_channel=1, channel_upsamplings=(16, 32, 64, 128, self.cond_dim))

        if per_frame_cond:
            # general embedding for per-frame cond
            self.sign_emb = nn.Linear(1, self.cond_dim)

        if per_frame_cond:
            self.cond_token_to_hidden = nn.Sequential(
                nn.LayerNorm(self.cond_dim),
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, time_dim)
                )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, padding_mode = self.padding_mode, groups = resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim = time_dim + int(self.cond_dim or 0) if self.cond_to_time == 'concat' else self.cond_dim)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads, cond_attention = self.cond_attention, cond_dim = self.cond_dim))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out, self.padding_mode) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads, cond_attention = self.cond_attention, cond_dim = self.cond_dim, per_frame_cond = per_frame_cond))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads, cond_attention = self.cond_attention, cond_dim = self.cond_dim))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in, self.padding_mode) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

        # conditional guidance
        self.null_text_token = nn.Parameter(torch.randn(1, self.cond_attention_tokens, self.cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, time_dim))

    def forward_with_guidance_scale(
        self,
        *args,
        **kwargs,
    ):

        guidance_scale = kwargs.pop('guidance_scale', 5.)

        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if guidance_scale == 1:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * guidance_scale

    def forward(
        self,
        x,
        time,
        cond = None,
        null_cond_prob = 0.,
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        batch, device = x.shape[0], x.device
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)
        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)
        r = x.clone()
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance
        batch, device = x.shape[0], x.device
        mask = prob_mask_like((batch,), null_cond_prob, device=device)

        if self.per_frame_cond:
            # add dimension to generate tokens per frame [batch x frames x 1] -> [batch x frames x embedding]
            cond = cond.unsqueeze(-1)
            # this acts on the last dimension, gives token embedding for attention
            label_emb_token = self.sign_emb(cond)
            # average over frames dim to get hidden embedding
            mean_pooled_text_tokens = label_emb_token.mean(dim = -2)
            # convert hidden averaged token embedding to hidden embedding
            label_emb_hidden = self.cond_token_to_hidden(mean_pooled_text_tokens)
        else:
            label_emb_hidden = self.sign_emb_CNN(cond)
            # generate tokens for cross-attention
            label_emb_token = None
            # generate tensor product for attention
            if self.cond_attention != 'none' and not self.cond_att_GRU:
                # generate tokens for cross-attention according to cond_attention_token                
                label_emb_token = repeat(label_emb_hidden, 'b x -> b n x', n = self.cond_attention_tokens)
            # leave embedding as is for GRU embedding
            elif self.cond_attention != 'none' and self.cond_att_GRU:
                label_emb_token = self.sign_emb_GRU(cond)

        if self.cond_attention != 'none':
            # null token
            label_mask_embed = rearrange(mask, 'b -> b 1 1')
            null_text_token = self.null_text_token.to(label_emb_token.dtype) # for some reason pytorch AMP not working
            # replace token by null token for masked samples
            label_emb_token = torch.where(label_mask_embed, null_text_token, label_emb_token)

        # null hidden
        label_mask_hidden = rearrange(mask, 'b -> b 1')
        null_text_hidden = self.null_text_hidden.to(label_emb_hidden.dtype) # for some reason pytorch AMP not working

        # replace hidden by null hidden for masked samples
        label_emb_hidden = torch.where(label_mask_hidden, null_text_hidden, label_emb_hidden)

        if self.cond_to_time == 'add':
            # add label embedding to time embedding
            t = t + label_emb_hidden
        elif self.cond_to_time == 'concat':
            t = torch.cat((t, label_emb_hidden), dim = -1)

        if self.use_temporal_attention_cond:
            label_emb_token_temporal = label_emb_token
        else:
            label_emb_token_temporal = None

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x, label_emb_mm = label_emb_token)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask, label_emb_mm = label_emb_token_temporal)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x, label_emb_mm = label_emb_token)
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask, label_emb_mm = label_emb_token_temporal)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x, label_emb_mm = label_emb_token)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask, label_emb_mm = label_emb_token_temporal)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)

# Gaussian Diffusion Trainer Class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        channels = 4,
        timesteps = 1000,
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9,
        sampling_timesteps = 1000,
        ddim_sampling_eta = 0.,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, guidance_scale = 1.):
        x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.forward_with_guidance_scale(x, t, cond = cond, guidance_scale = guidance_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond = None, clip_denoised = True, guidance_scale = 1.):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, cond = cond, guidance_scale = guidance_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, guidance_scale = 1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, guidance_scale = guidance_scale)

        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, cond = None, batch_size = 16, guidance_scale = 1.):
        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, num_frames, image_size, image_size), cond = cond, guidance_scale = guidance_scale)

    @torch.inference_mode()
    def ddim_sample(self, shape, cond = None, guidance_scale = 1.):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'DDIM sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise = self.denoise_fn.forward_with_guidance_scale(img, time_cond, cond = cond, guidance_scale = guidance_scale)
            x_start = self.predict_start_from_noise(img, time_cond, noise = pred_noise)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return unnormalize_img(img)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond = None, noise = None, **kwargs):

        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(x_noisy, t, cond = cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c = self.channels, f = self.num_frames, h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = normalize_img(x)
        return self.p_losses(x, t, *args, **kwargs)

# trainer class

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif
def video_tensor_to_gif(tensor, path, duration = 200, loop = 0, optimize = False):      # NOTE changed optimize to False 
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    # convert images since optimize = False gives issues with non-Palette images
    if optimize == False:
        images = map(lambda img: img.convert('L').convert('P'), images)
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor
def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f)) # (since pad starts from the last dim)

class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        labels_scaling = None,
        selected_channels = [0, 1, 2, 3],
        num_frames = 16,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif'],
        per_frame_cond = False,
        reference_frame = 'eulerian',
    ):
        super().__init__()
        self.image_size = image_size
        self.selected_channels = selected_channels
        self.num_frames = num_frames

        # load topo data
        topo_folder = folder + 'gifs/topo/'
        self.paths_top = [p for ext in exts for p in Path(f'{topo_folder}').glob(f'**/*.{ext}')]
        # sort paths by number of name
        self.paths_top = sorted(self.paths_top, key=lambda x: int(x.name.split('.')[0]))
        assert all([int(p.stem) == i for i, p in enumerate(self.paths_top)]), 'file position is not equal to index'

        if reference_frame == 'lagrangian':
            # load disp_x data
            disp_x_folder = folder + 'gifs/disp_x/'
            self.paths_disp_x = [p for ext in exts for p in Path(f'{disp_x_folder}').glob(f'**/*.{ext}')]
            # sort paths by number of name
            self.paths_disp_x = sorted(self.paths_disp_x, key=lambda x: int(x.name.split('.')[0]))
            assert all([int(p.stem) == i for i, p in enumerate(self.paths_disp_x)]), 'file position is not equal to index'

            assert len(self.paths_disp_x) == len(self.paths_top), 'number of files in fields and top folders are not equal.'

            # load disp_y data
            disp_y_folder = folder + 'gifs/disp_y/'
            self.paths_disp_y = [p for ext in exts for p in Path(f'{disp_y_folder}').glob(f'**/*.{ext}')]
            # sort paths by number of name
            self.paths_disp_y = sorted(self.paths_disp_y, key=lambda x: int(x.name.split('.')[0]))
            assert all([int(p.stem) == i for i, p in enumerate(self.paths_disp_y)]), 'file position is not equal to index'
            
            assert len(self.paths_disp_y) == len(self.paths_top), 'number of files in fields and top folders are not equal.'
                
        # load mises data
        mises_folder = folder + 'gifs/mises/'
        self.paths_mises = [p for ext in exts for p in Path(f'{mises_folder}').glob(f'**/*.{ext}')]
        # sort paths by number of name
        self.paths_mises = sorted(self.paths_mises, key=lambda x: int(x.name.split('.')[0]))
        assert all([int(p.stem) == i for i, p in enumerate(self.paths_mises)]), 'file position is not equal to index'

        assert len(self.paths_mises) == len(self.paths_top), 'number of files in fields and top folders are not equal.'

        # load s_22 data
        s_22_folder = folder + 'gifs/s_y/'
        self.paths_s_22 = [p for ext in exts for p in Path(f'{s_22_folder}').glob(f'**/*.{ext}')]
        # sort paths by number of name
        self.paths_s_22 = sorted(self.paths_s_22, key=lambda x: int(x.name.split('.')[0]))
        assert all([int(p.stem) == i for i, p in enumerate(self.paths_s_22)]), 'file position is not equal to index'

        assert len(self.paths_s_22) == len(self.paths_top), 'number of files in fields and top folders are not equal.'

        # load ener data
        ener_folder = folder + 'gifs/ener/'
        self.paths_ener = [p for ext in exts for p in Path(f'{ener_folder}').glob(f'**/*.{ext}')]
        # sort paths by number of name
        self.paths_ener = sorted(self.paths_ener, key=lambda x: int(x.name.split('.')[0]))
        assert all([int(p.stem) == i for i, p in enumerate(self.paths_ener)]), 'file position is not equal to index'

        assert len(self.paths_ener) == len(self.paths_top), 'number of files in fields and top folders are not equal.'

        frame_range_file = folder + 'cleaned_frame_range_data.csv'
        # we manually apply a 'global-min-max-1'-scaling to the gifs, for which we need the original min/max values
        self.frame_ranges = torch.tensor(np.genfromtxt(frame_range_file, delimiter=','))

        if reference_frame == 'eulerian':
            self.max_von_Mises = torch.max(self.frame_ranges[:,0])
            self.min_s_22 = torch.min(self.frame_ranges[:,1])
            self.max_s_22 = torch.max(self.frame_ranges[:,2])
            self.max_strain_energy = torch.max(self.frame_ranges[:,3])

            self.zero_disp_y = None
        elif reference_frame == 'lagrangian':
            self.min_disp_x = torch.min(self.frame_ranges[:,0])
            self.max_disp_x = torch.max(self.frame_ranges[:,1])
            self.min_disp_y = torch.min(self.frame_ranges[:,2])
            self.max_disp_y = torch.max(self.frame_ranges[:,3])
            self.max_von_Mises = torch.max(self.frame_ranges[:,4])
            self.min_s_22 = torch.min(self.frame_ranges[:,5])
            self.max_s_22 = torch.max(self.frame_ranges[:,6])
            self.max_strain_energy = torch.max(self.frame_ranges[:,7])

            self.zero_disp_y = self.normalize(torch.zeros(1), self.min_disp_y, self.max_disp_y)

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

        label_file = folder + 'cleaned_stress_strain_data.csv'

        labels_np = np.genfromtxt(label_file, delimiter=',')
        if per_frame_cond:
            strain = 0.2
            # interpolate stress data to match number of frames
            given_points = np.linspace(0., strain, num = labels_np.shape[1])
            eval_points = np.linspace(0., strain, num = num_frames)
            # overwrite first eval point since we take first frame at 1% strain
            eval_points[0] = 0.01*strain
            # interpolate stress data to match number of frames for full array
            labels_np = np.array([np.interp(eval_points, given_points, labels_np[i,:]) for i in range(labels_np.shape[0])])
            self.labels = torch.tensor(labels_np).float()
        else:
        # NOTE Remove first label index since only contains zeros, keep in mind that last value of simulation was already removed in preprocessing.
            self.labels = torch.tensor(labels_np[:,1:]).float()
        self.detached_labels = self.labels.clone().detach().numpy()

        # compute normalization if not given
        if labels_scaling is None:
            # normalize labels to [-1, 1] based on global min/max (i.e., min/max of all samples in training set)
            self.labels_scaling = Normalization(self.labels, ['continuous']*self.labels.shape[1], 'global-min-max-2')
        else:
            # use given normalization (relevant for validation set, which should use same normalization as training set)
            self.labels_scaling = labels_scaling
        # apply normalization
        self.labels = self.labels_scaling.normalize(self.labels)

        self.reference_frame = reference_frame

    def interpolate(self, tensor, num_frames):
        f = tensor.shape[1]
        if f == num_frames:
            return tensor
        if f > num_frames:
            return tensor[:, :num_frames]
        return F.interpolate(tensor.unsqueeze(0), num_frames).squeeze(0)

    def normalize(self, arr, min_val, max_val):
        return (arr - min_val) / (max_val - min_val)

    def unnorm(self, arr, min_val, max_val):
        return arr * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.paths_top)

    def __getitem__(self, index):

        if self.reference_frame == 'eulerian':
            paths_top = self.paths_top[index]
            paths_mises = self.paths_mises[index]
            paths_s_22 = self.paths_s_22[index]
            paths_ener = self.paths_ener[index]

            tensor = torch.cat((gif_to_tensor(paths_top, channels=1, transform = self.transform), 
                                gif_to_tensor(paths_mises, channels=1, transform = self.transform),
                                gif_to_tensor(paths_s_22, channels=1, transform = self.transform),
                                gif_to_tensor(paths_ener, channels=1, transform = self.transform),
                                ), dim=0)
            
            ## convert tensor to [0,1]-normalized global range
            # unnormalize first
            tensor[1,:,:,:] = self.unnorm(tensor[1,:,:,:], 0., self.frame_ranges[index,0])
            tensor[2,:,:,:] = self.unnorm(tensor[2,:,:,:], self.frame_ranges[index,1], self.frame_ranges[index,2])
            tensor[3,:,:,:] = self.unnorm(tensor[3,:,:,:], 0., self.frame_ranges[index,3])

            # set values to zero for all pixels where topology is zero
            # IMPORTANT: we must do this after scaling values true range to ensure a 0 value corresponds to true 0 field value
            for i in range(1,4):
                tensor[i,:,:,:][topologies[0,:,:,:] == 0.] = 0.

            # normalize to global range
            tensor[1,:,:,:] = self.normalize(tensor[1,:,:,:], 0., self.max_von_Mises)
            tensor[2,:,:,:] = self.normalize(tensor[2,:,:,:], self.min_s_22, self.max_s_22)
            tensor[3,:,:,:] = self.normalize(tensor[3,:,:,:], 0., self.max_strain_energy)

        elif self.reference_frame == 'lagrangian' and self.num_frames != 1:
            paths_top = self.paths_top[index]
            paths_disp_x = self.paths_disp_x[index]
            paths_disp_y = self.paths_disp_y[index]
            paths_mises = self.paths_mises[index]
            paths_s_22 = self.paths_s_22[index]

            topologies = gif_to_tensor(paths_top, channels=1, transform = self.transform)

            tensor = torch.cat((gif_to_tensor(paths_disp_x, channels=1, transform = self.transform), 
                                gif_to_tensor(paths_disp_y, channels=1, transform = self.transform),
                                gif_to_tensor(paths_mises, channels=1, transform = self.transform),
                                gif_to_tensor(paths_s_22, channels=1, transform = self.transform),
                                ), dim=0)
            
            ## convert tensor to [0,1]-normalized global range
            # unnormalize first
            tensor[0,:,:,:] = self.unnorm(tensor[0,:,:,:], self.frame_ranges[index,0], self.frame_ranges[index,1])
            tensor[1,:,:,:] = self.unnorm(tensor[1,:,:,:], self.frame_ranges[index,2], self.frame_ranges[index,3])
            tensor[2,:,:,:] = self.unnorm(tensor[2,:,:,:], 0., self.frame_ranges[index,4])
            tensor[3,:,:,:] = self.unnorm(tensor[3,:,:,:], self.frame_ranges[index,5], self.frame_ranges[index,6])

            # set values to zero for all pixels where topology is zero
            # IMPORTANT: we must do this after scaling values true range to ensure a 0 value corresponds to true 0 field value
            for i in range(4):
                tensor[i,:,:,:][topologies[0,:,:,:] == 0.] = 0.

            # normalize to global range
            tensor[0,:,:,:] = self.normalize(tensor[0,:,:,:], self.min_disp_x, self.max_disp_x)
            tensor[1,:,:,:] = self.normalize(tensor[1,:,:,:], self.min_disp_y, self.max_disp_y)
            tensor[2,:,:,:] = self.normalize(tensor[2,:,:,:], 0., self.max_von_Mises)
            tensor[3,:,:,:] = self.normalize(tensor[3,:,:,:], self.min_s_22, self.max_s_22)

        # only relevant for ablation study, where we consider two channels (topology and sigma_22)
        elif self.reference_frame == 'lagrangian' and self.num_frames == 1:
            paths_top = self.paths_top[index]
            paths_mises = self.paths_mises[index]
            paths_s_22 = self.paths_s_22[index]

            topologies = gif_to_tensor(paths_top, channels=1, transform = self.transform)

            tensor = torch.cat((gif_to_tensor(paths_top, channels=1, transform = self.transform),
                                gif_to_tensor(paths_s_22, channels=1, transform = self.transform),
                                ), dim=0)
            
            ## convert tensor to [0,1]-normalized global range
            # unnormalize first
            tensor[1,:,:,:] = self.unnorm(tensor[1,:,:,:], self.frame_ranges[index,5], self.frame_ranges[index,6])

            # set values to zero for all pixels where topology is zero
            # IMPORTANT: we must do this after scaling values true range to ensure a 0 value corresponds to true 0 field value
            tensor[1,:,:,:][topologies[0,:,:,:] == 0.] = 0.

            # normalize to global range
            tensor[1,:,:,:] = self.normalize(tensor[1,:,:,:], self.min_s_22, self.max_s_22)

            self.selected_channels = [0,1]

        tensor = tensor[self.selected_channels,:,:,:]
        labels = self.labels[index,:]

        return self.cast_num_frames_fn(tensor), labels

# Trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        validation_folder,
        selected_channels,
        *,
        ema_decay = 0.995,
        train_batch_size = 4,
        test_batch_size = 2,
        train_lr = 1.e-4,
        train_num_steps = 100000,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './',
        max_grad_norm = None,
        log = True,
        null_cond_prob = 0.,
        per_frame_cond = False,
        reference_frame = 'eulerian',
        run_name = None,
        accelerator = None,
        wandb_username = None
    ):
        super().__init__()

        self.accelerator = accelerator

        if log:
            self.accelerator.init_trackers(
                project_name='metamaterial_diffusion',
                init_kwargs={
                    'wandb': {
                        'name': run_name,
                        'entity': wandb_username,
                    }
                },
            )
            self.log_fn = self.accelerator.log
        else:
            self.log_fn = noop

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.step = 0

        self.model = self.accelerator.prepare(diffusion_model)

        self.device = self.accelerator.device
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.test_batch_size = test_batch_size // 2 # since evaluation requires more memory
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        num_frames = diffusion_model.num_frames
        self.num_frames = num_frames

        self.selected_channels = selected_channels

        self.ds = Dataset(folder, image_size, labels_scaling = None, selected_channels=self.selected_channels, \
            num_frames = num_frames, per_frame_cond = per_frame_cond, reference_frame=reference_frame)
        self.dl = cycle(self.accelerator.prepare(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True)))

        self.accelerator.print(f'found {len(self.ds)} videos as gif files in {folder}')
        assert len(self.ds) > 0, 'could not find any gif files in folder'

        # create test set, use same normalization as for training set, i.e., same frame range and label scaling
        self.ds_test = Dataset(validation_folder, image_size, labels_scaling=self.ds.labels_scaling, \
            selected_channels=self.selected_channels, num_frames = num_frames, per_frame_cond = per_frame_cond, reference_frame=reference_frame)
        self.dl_test = self.accelerator.prepare(data.DataLoader(self.ds_test, batch_size = self.test_batch_size, shuffle=False, pin_memory=True))

        self.opt = self.accelerator.prepare(Adam(self.model.parameters(), lr = train_lr))

        self.max_grad_norm = max_grad_norm

        self.null_cond_prob = null_cond_prob

        self.per_frame_cond = per_frame_cond

        self.reset_parameters()

        # obtain number of processes
        self.num_processes = self.accelerator.num_processes

        self.folder = folder

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def cond_to_gpu(
        self,
        cond,
        num_samples
    ):
        # obtain GPU indices
        gpu_index = self.accelerator.process_index
        # obtain share of test_cond from test_cond_full based on GPU index in continuous blocks
        tot_preds = len(cond) * num_samples
        preds_per_gpu = tot_preds // self.accelerator.num_processes
        # perform the slicing for each process
        start_idx = gpu_index * preds_per_gpu
        end_idx = (gpu_index + 1) * preds_per_gpu if gpu_index != self.accelerator.num_processes - 1 else cond.shape[0]
        gpu_cond = cond[start_idx:end_idx, :]
        # sample from model
        local_num_samples = gpu_cond.shape[0]
        batches = num_to_groups(local_num_samples, self.test_batch_size)
        # create list of indices for each batch
        indices = []
        start = 0
        for batch_size in batches:
            end = start + batch_size
            indices.append((start, end))
            start = end
        # split test_cond into smaller tensors using the indices
        batched_gpu_cond = []
        for i, j in indices:
            batched_gpu_cond.append(gpu_cond[i:j, :])
        return batched_gpu_cond

    def save(
        self,
        step = None
    ):

        if step == None:
            step = self.step
        save_dir = str(self.results_folder) + '/model/step_' + str(step)
        if self.accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
        save_dir = save_dir + '/checkpoint.pt'

        self.accelerator.wait_for_everyone()

        save_obj = dict(
            model = self.model.state_dict(),
            optimizer = self.opt.state_dict(),
            steps = self.step,
        )

        if self.ema_model is not None:
            save_obj = {**save_obj, 'ema': self.ema_model.state_dict()}

        # save to path
        with open(save_dir, 'wb') as f:
            torch.save(save_obj, f)

        self.accelerator.print(f'\ncheckpoint saved to {save_dir}')

    def load(
        self,
        step,
        strict = True,
    ):
        path = str(self.results_folder) + '/model/step_' + str(step) + '/checkpoint.pt'

        if not os.path.isfile(path):
            raise FileNotFoundError(f'trainer checkpoint not found at {str(path)}. Please check path or run load_model_step = None')

        # to avoid extra GPU memory usage in main process when using Accelerate
        with open(path, 'rb') as f:
            loaded_obj = torch.load(f, map_location='cpu')

        try:
            self.model.load_state_dict(loaded_obj['model'], strict = strict)
        except RuntimeError:
            print("Failed loading state dict.")

        self.step = (loaded_obj['steps'])

        try:
            self.opt.load_state_dict(loaded_obj['optimizer'])
        except:
            self.accelerator.print('could not load optimizer and scaler, possibly because you have turned on mixed precision training since the last run. resuming with new optimizer and scalers')

        try:
            self.ema_model.load_state_dict(loaded_obj['ema'], strict = strict)
        except RuntimeError:
            print("Failed loading state dict.d")

        self.accelerator.print(f'checkpoint loaded from {path}')
        return loaded_obj

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        load_model_step = None,
        num_samples = 1,
        num_preds = 1
    ):
        assert callable(self.log_fn)

        # trying to load model trained to given step
        if load_model_step is not None:
            self.load(step=load_model_step)
            # increment step counter since training starts with the next step (for consistency)
            self.step += 1

        if self.accelerator.is_main_process:
            start_time = time.time() # start timer for tracking purposes

        while self.step <= self.train_num_steps:
            # accumulate context manager (for gradient accumulation)
            data, cond = next(self.dl)
            with self.accelerator.accumulate(self.model):
                self.opt.zero_grad()
                loss = self.model(
                    x = data,
                    cond = cond,
                    null_cond_prob = self.null_cond_prob,
                    prob_focus_present = prob_focus_present,
                    focus_present_mask = focus_present_mask,
                )
                self.accelerator.backward(loss)
                # gradient clipping
                if self.accelerator.sync_gradients and exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

            self.log_fn({'training loss': loss.item()}, step = self.step)

            if self.step % self.update_ema_every == 0:
                self.accelerator.wait_for_everyone()
                self.step_ema()

            if (self.step > 0 and self.step % self.save_and_sample_every == 0) or self.step == self.train_num_steps:

                # verify that all processes have the same step (just to be sure)
                self.accelerator.wait_for_everyone()
                gathered_steps = self.accelerator.gather(torch.tensor(self.step).to(self.accelerator.device))
                if gathered_steps.numel() > 1:
                    assert torch.all(gathered_steps == gathered_steps[0])
                # print current step and total time elapsed in hours, minutes and seconds
                if self.accelerator.is_main_process:
                    cur_time = time.time()
                    elapsed_time = cur_time - start_time
                    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                    print(f'current step: {self.step}, total time elapsed: {elapsed_time}')
                # evaluate network on validation set (including Abaqus simulations)
                self.eval_network(prob_focus_present, focus_present_mask, num_samples = num_samples, num_preds = num_preds)
                if self.accelerator.is_main_process:
                    elapsed_time_validation = time.time() - cur_time
                    elapsed_time_validation = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_validation))
                    print(f'time elapsed for validation: {elapsed_time_validation}')
            
            if self.step != self.train_num_steps:
                self.step += 1
            else:
                break
            
        self.accelerator.print('Training completed.')
            
        # save model at the end of the training
        self.accelerator.wait_for_everyone()
        self.save(step=self.step)

        self.accelerator.print('Evaluate target data.')
        target_labels_dir = self.folder + './data/target_responses.csv'

        # eval on target data (decrement step since we consider the last step to be the one before the training ended)
        self.eval_target(target_labels_dir)

        # end training
        self.accelerator.end_training()

    def eval_network(
        self,
        prob_focus_present,
        focus_present_mask,
        guidance_scale = 5.,
        num_samples = 1,
        num_preds = 1
    ):

        mode = 'training'

        if self.accelerator.is_main_process:
            # create folder for each milestone
            os.makedirs('./' + str(self.results_folder) + '/' + mode + '/step_' + str(self.step) + '/gifs', exist_ok=True)

            losses = []
            # select random conditionings from validation set to sample videos
            req_idcs = int(np.ceil(num_samples / self.test_batch_size))
            rand_idcs = np.random.choice(len(self.dl_test), req_idcs, replace=False)
            test_cond_list = []

        for idx, (data, cond) in enumerate(self.dl_test):
            loss = self.model(
                x = data,
                cond = cond,
                null_cond_prob = self.null_cond_prob,
                prob_focus_present = prob_focus_present,
                focus_present_mask = focus_present_mask,
            )
            if idx in rand_idcs and self.accelerator.is_main_process:
                test_cond_list.append(cond.clone().detach())
            all_losses = self.accelerator.gather_for_metrics(loss)
            if self.accelerator.is_main_process:
                loss = torch.mean(all_losses)
                losses.append(loss.item())

        # compute validation loss over full test data
        if self.accelerator.is_main_process:
            test_loss = np.mean(losses)
            self.log_fn({'validation loss': test_loss}, step = self.step)

            if num_samples > 0:
                # sample from model conditioned on randomly selected test batch
                test_cond = torch.cat(test_cond_list, dim=0)[:num_samples,:]
                # repeat each value of test_cond self.num_samples times
                test_cond_full_repeated = test_cond.repeat_interleave(num_preds, dim=0)
            else:
                test_cond_full_repeated = None

        # broadcast test_cond_full_repeated across GPUs
        self.accelerator.wait_for_everyone()
        test_cond_full_repeated = broadcast_object_list([test_cond_full_repeated])[0]

        if test_cond_full_repeated is not None:
            # distribute test_cond_full_repeated across available GPUs
            batched_test_cond = self.cond_to_gpu(test_cond_full_repeated, num_samples)
            # generate samples using each split of test_cond
            ema_model = self.accelerator.unwrap_model(self.ema_model)
            all_videos_list = []
            for cond in batched_test_cond:
                samples = ema_model.sample(cond=cond, guidance_scale = guidance_scale)
                all_videos_list.append(samples)
            # concatenate the generated samples into a single tensor
            all_videos_list = torch.cat(all_videos_list, dim = 0)
            # gather all samples from all processes
            self.accelerator.wait_for_everyone()
            all_videos_list = self.accelerator.gather(all_videos_list)
            test_cond = self.accelerator.gather(test_cond)
            # do further evaluation on main process
            if self.accelerator.is_main_process:
                self.save_preds(all_videos_list, num_samples = num_samples)

    def eval_target(
        self,
        target_labels_dir,
        guidance_scale = 5.,
        num_preds = 1
    ):
        assert callable(self.log_fn)

        mode = 'eval_target_w_' + str(guidance_scale)

        if self.accelerator.is_main_process:
            # create folder for prediction
            eval_idx = 0
            while os.path.exists('./' + str(self.results_folder) + '/' + mode + '_' + str(eval_idx) + '/step_' + str(self.step)):
                eval_idx += 1
            mode = mode + '_' + str(eval_idx)
            
            os.makedirs('./' + str(self.results_folder) + '/' + mode + '/step_' + str(self.step) + '/gifs', exist_ok=True)

            # convert target_labels_dir to tensor
            try:
                target_labels = np.genfromtxt(target_labels_dir, delimiter=',')
            except:
                self.accelerator.print('Could not load target labels.')
                return

            if len(target_labels.shape) == 1:
                target_labels = target_labels[np.newaxis,:]

            if self.per_frame_cond:
                if self.num_frames != target_labels.shape[1]:
                    strain = 0.2
                    # interpolate stress data to match number of frames
                    given_points = np.linspace(0., strain, num = target_labels.shape[1])
                    eval_points = np.linspace(0., strain, num = self.num_frames)
                    # overwrite first eval point since we take first frame at 1% strain
                    eval_points[0] = 0.01*strain
                    # interpolate stress data to match number of frames for full array
                    target_labels_red = np.array([np.interp(eval_points, given_points, target_labels[i,:]) for i in range(target_labels.shape[0])])
                    target_labels_red = torch.tensor(target_labels_red).float().to(self.device)
                    test_cond_full = target_labels_red
                else:
                    test_cond_full = torch.tensor(target_labels).float().to(self.device)
            else:
                # NOTE Remove first label index since only contains zeros (we do not have to remove last value since we only pass 51 values)
                test_cond_full = torch.tensor(target_labels[:,1:]).float().to(self.device)

            # normalize target_labels
            test_cond_full = self.ds.labels_scaling.normalize(test_cond_full)

            num_samples = len(test_cond_full)

            # repeat each value of test_cond self.red_preds_per_sample times
            test_cond_full_repeated = test_cond_full.repeat_interleave(num_preds, dim=0)

        self.accelerator.wait_for_everyone()

        # braodcast test_cond_full_repeated across GPUs
        test_cond_full_repeated = broadcast_object_list([test_cond_full_repeated])[0]

        # distribute test_cond_full_repeated across available GPUs
        batched_test_cond = self.cond_to_gpu(test_cond_full_repeated, num_samples)

        # generate samples using each split of test_cond
        ema_model = self.accelerator.unwrap_model(self.ema_model)
        all_videos_list = []
        for cond in batched_test_cond:
            samples = ema_model.sample(cond=cond, guidance_scale = guidance_scale)
            all_videos_list.append(samples)

        # concatenate the generated samples into a single tensor
        all_videos_list = torch.cat(all_videos_list, dim = 0)

        # gather all samples from all processes
        self.accelerator.wait_for_everyone()

        # pad across processes since gather needs tensors of equal length
        padded_all_videos_list = self.accelerator.pad_across_processes(all_videos_list, dim=0)

        gathered_all_videos_list = self.accelerator.gather(padded_all_videos_list)

        # gather the lengths of the original all_videos_list tensors
        original_lengths = self.accelerator.gather(torch.tensor(all_videos_list.shape[0]).to(all_videos_list.device))

        # do further evaluation on main process
        if self.accelerator.is_main_process:

            # Remove padding from the gathered tensor
            unpadded_all_videos_list = []
            max_length = padded_all_videos_list.shape[0]
            start_idx = 0
            for length in original_lengths:
                end_idx = start_idx + length
                unpadded_all_videos_list.append(gathered_all_videos_list[start_idx:end_idx])
                start_idx += max_length

            gathered_all_videos = torch.cat(unpadded_all_videos_list, dim = 0)

            self.save_preds(gathered_all_videos, num_samples=num_samples, mode=mode)

    def save_preds(
        self,
        pred_videos,
        num_samples,
        mode='training'
    ):

        # save predictions to gifs
        padded_pred_videos = F.pad(pred_videos, (2, 2, 2, 2))
        one_gif = rearrange(padded_pred_videos, '(i j) c f h w -> c f (i h) (j w)', i = num_samples)

        for j, pred_channel in enumerate(self.selected_channels):
            video_path = './' + str(self.results_folder) + '/' + mode + '/step_' + str(self.step) + '/gifs/pred_' + str(pred_channel+1) + '.gif'
            video_tensor_to_gif(one_gif[None, j], video_path)