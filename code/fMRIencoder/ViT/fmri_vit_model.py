import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sc_mbm.utils as ut

class PatchEmbed1D(nn.Module):
    
    def __init__(self, num_voxels=5936, patch_size=16, in_chans=1, embed_dim=1024, norm_layer=None):
        super().__init__()

        #num_patches = num_voxels // patch_size
        #self.patch_shape = patch_size
        #self.num_voxels = num_voxels
        #self.patch_size = patch_size
        #self.num_patches = num_patches

        self.num_voxels = num_voxels
        self.patch_size = patch_size
        self.patches_resolution = in_chans
        self.num_patches = num_voxels // patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        
    def forward(self, x,):
        B, C, L = x.shape # batch, channel, num_voxels

        x = self.proj(x).transpose(1, 2).contiguous() # B x num_patches x 1024 

        if self.norm is not None:
            x = self.norm(x)

        return x


# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

#import torch
#import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

#try:
#    import os, sys

##    kernel_path = os.path.abspath(os.path.join('..'))
#    sys.path.append(kernel_path)
#    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

#except:
#    WindowProcess = None
#    WindowProcessReverse = None
#    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, N, L = x.shape
    x = x.view(B, N // window_size, window_size, L)
    windows = x.view(-1, window_size, L)
    return windows


def window_reverse(windows, window_size, N):

    """
    Args:
        windows: (num_windows*B, window_size, L)
        window_size (int): Window size
        C (int): number of channel of signal
        

    Returns:
        x: (B, C, L)
    """
    B = int(windows.shape[0] / (N / window_size)) # B*nW / nW = B
    x = windows.view(B, N // window_size, window_size, -1)
    x = x.view(B, N, -1)

    return x

def pad_to_patch_size(x, patch_size): # patch ㅋ크기에 딱 맞아 떠러지도록 padding 
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length): # patch 수가 동일하도록 padding
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x
    
    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, L = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, L // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q,k,v.shape:  B_, num_heads, N, L//num_heads
        q = q * self.scale # scale ????
        attn = (q @ k.transpose(-2, -1)) # B_, num_heads, N, N
        

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, L)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, lobe_wise=False, decoder_tf = False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.lobe_wise = lobe_wise
        self.decoder_tf = decoder_tf
        self.shift_size = 0

        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp4 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp5 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp6 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp7 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        
        attn_mask = None



        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        #H, W = self.input_resolution
        
        #assert L == H * W, "input feature has wrong size"
        
        roi_mlp = False

        #cls_token = x[:,:1,:]
        #x = x[:,1:,:]
        B, N, L = x.shape

        shortcut = x
        x = self.norm1(x)

        if self.decoder_tf: ## Decoder의 경우 mask가 합쳐지고 원래 patch자리로 돌아가기 때문에 window size 다시 정해줘야 함.
            
            # partition windows
            x_windows = window_partition(x, self.window_size)  # nW*B, window_size, L  nW->7, window_size ->53*(1-.075)=13

            #x_windows = x_windows.view(-1, self.window_size, L)  # nW*B, window_size, L

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, L

            x = window_reverse(attn_windows, self.window_size, N)  # B N' L

            x = shortcut + self.drop_path(x)

            #x = torch.cat((cls_token,x), dim=1)

            # FFN
            if roi_mlp:
                x_ = self.norm2(x)
                x_ = list(torch.split(x_, x_.shape[1]//7, dim=1))
                x_[0] = self.mlp1(x_[0])
                x_[1] = self.mlp2(x_[1])
                x_[2] = self.mlp3(x_[2])
                x_[3] = self.mlp4(x_[3])
                x_[4] = self.mlp5(x_[4])
                x_[5] = self.mlp6(x_[5])
                x_[6] = self.mlp7(x_[6])
                
                x = torch.cat([x_[i] for i in range(7)], dim=1)
                x = x + self.drop_path(x)
            
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x
        
        else: ### encoder 경우 mask 되지 않는 애들만 사용하므로 window size 다시 정해줘야 함.
            # partition windows
            x_windows = window_partition(x, self.window_size)  # nW*B, window_size, L

            x_windows = x_windows.view(-1, self.window_size, L)  # nW*B, window_size, L

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, L

            x = window_reverse(attn_windows, self.window_size, N)  # B N' L

            x = shortcut + self.drop_path(x)

            #x = torch.cat((cls_token,x), dim=1)

            # FFN
            
            if roi_mlp:
                x_ = self.norm2(x)
                x_ = list(torch.split(x_, x_.shape[1]//7, dim=1))
                x_[0] = self.mlp1(x_[0])
                x_[1] = self.mlp2(x_[1])
                x_[2] = self.mlp3(x_[2])
                x_[3] = self.mlp4(x_[3])
                x_[4] = self.mlp5(x_[4])
                x_[5] = self.mlp6(x_[5])
                x_[6] = self.mlp7(x_[6])
                
                x = torch.cat([x_[i] for i in range(7)], dim=1)
      
                x = x + self.drop_path(x)
            
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

'''
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
'''

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 fused_window_process=False, lobe_wise=False,decoder_tf=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        # build blocks
        # 동일 stage에서의 transformer 반복
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path,
                                 act_layer=nn.GELU,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 lobe_wise=lobe_wise, decoder_tf=decoder_tf)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
 
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (list): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, eeg_lengths=128, patch_size=12, in_chans=28, num_classes=2,
                 embed_dim=768, depths=[ 2, 2, 2, 2], num_heads=[4, 4, 4, 4],
                 window_sizes=[4, 14,28], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, lobe_wise=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        ## *** check 
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed1D(
            eeg_lengths=eeg_lengths, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

       

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        #x = self.norm(x)  # B C L
        #x = self.avgpool(x.transpose(1, 2))  # B L 1
        #x = torch.flatten(x, 1)
        return x


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
       

class MAEforFMRI(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, num_voxels=53*16*7, patch_size=16, embed_dim=1024, 
                 in_chans=1, window_size=53,
                 depth=24, num_heads=16, 
                 decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 mask_ratio=0.75):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(num_voxels, patch_size, in_chans, embed_dim, norm_layer=None)
        self.mask_ratio = mask_ratio
        
     

        num_patches = self.patch_embed.num_patches # 53*7=371
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.window_size = window_size
        encoder_window_size = int(window_size * (1-mask_ratio))
        decoder_window_size = window_size
        self.encoder_window_size = encoder_window_size
        
        self.blocks = nn.ModuleList([
            BasicLayer(dim=embed_dim, input_resolution=num_patches, num_heads=num_heads, window_size=encoder_window_size, mlp_ratio=mlp_ratio, qkv_bias=True, 
                                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,  norm_layer=norm_layer,
                                 use_checkpoint=False,
                                 fused_window_process=False,lobe_wise=False, decoder_tf=False)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            BasicLayer(dim=decoder_embed_dim, input_resolution=num_patches, num_heads=decoder_num_heads, window_size=decoder_window_size, mlp_ratio=mlp_ratio, qkv_bias=True, 
                                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=norm_layer,
                                 use_checkpoint=False,
                                 fused_window_process=False,lobe_wise=False, decoder_tf=True)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True) # encoder to decoder
    
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        
   
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    
    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        x: (N, L, patch_size)
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_embed.patch_size
        h = x.shape[1]
        
        imgs = x.reshape(shape=(x.shape[0], 1, h * p))
        return imgs
    
    def roi_shuffle_index(self, x, roi_len, roi_mask_ratio, N, roi_name):
        
        roi_keep_len = int(roi_len*(1-roi_mask_ratio))
        if roi_name == 'v1d':
            roi_shuffle = torch.tensor([-70]*roi_keep_len + [10]*(roi_len-roi_keep_len), device=x.device).repeat(N,1) # keep = -10, mask = 10
            roi_shuffle = roi_shuffle+torch.rand(N, roi_len, device=x.device) # mask

        elif roi_name == 'v1v':
            roi_shuffle = torch.tensor([-60]*roi_keep_len + [20]*(roi_len-roi_keep_len), device=x.device).repeat(N,1)
            roi_shuffle = roi_shuffle+torch.rand(N, roi_len, device=x.device)
        
        elif roi_name == 'v2d':
            roi_shuffle = torch.tensor([-50]*roi_keep_len + [30]*(roi_len-roi_keep_len), device=x.device).repeat(N,1)
            roi_shuffle = roi_shuffle+torch.rand(N, roi_len, device=x.device)

        elif roi_name == 'v2v':
            roi_shuffle = torch.tensor([-40]*roi_keep_len + [40]*(roi_len-roi_keep_len), device=x.device).repeat(N,1)
            roi_shuffle = roi_shuffle+torch.rand(N, roi_len, device=x.device)

        elif roi_name == 'v3d':
            roi_shuffle = torch.tensor([-30]*roi_keep_len + [50]*(roi_len-roi_keep_len), device=x.device).repeat(N,1)
            roi_shuffle = roi_shuffle+torch.rand(N, roi_len, device=x.device)

        elif roi_name == 'v3v':
            roi_shuffle = torch.tensor([-20]*roi_keep_len + [60]*(roi_len-roi_keep_len), device=x.device).repeat(N,1)
            roi_shuffle = roi_shuffle+torch.rand(N, roi_len, device=x.device)

        else:
            roi_shuffle = torch.tensor([-10]*roi_keep_len + [70]*(roi_len-roi_keep_len), device=x.device).repeat(N,1)
            roi_shuffle = roi_shuffle+torch.rand(N, roi_len, device=x.device)
            
        roi_shuffle.to(x.device)
        
        for i in range(N):
            roi_shuffle[i] = roi_shuffle[i,torch.randperm(roi_len)]
            # 배치 별로 random하게 섞기,앞의 voxel만 mask되지 않도록 하기 위함

        # roi에 따라 높은 값으로 mask 하여, 순서대로 재배치하였을 때, roi끼리 뭉쳐서 배치되도록 설정
        # mak는 양수 임으로 나열하였을 때, mak된 애들은 가장 뒤족에 나열된다.

        return roi_shuffle, roi_keep_len

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        Roi-wise stratify random_masking
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, nPatch, dim
      
        v1d_roi_len = 53
        v1v_roi_len = 53
        v2d_roi_len = 53
        v2v_roi_len = 53
        v3d_roi_len = 53
        v3v_roi_len = 53
        v4_roi_len = 53
                
        v1d_shuffle_index, v1d_keep_len = self.roi_shuffle_index(x,v1d_roi_len,  mask_ratio, N, 'v1d')
        v1v_shuffle_index, v1v_keep_len = self.roi_shuffle_index(x,v1v_roi_len, mask_ratio, N, 'v1v')
        v2d_shuffle_index, v2d_keep_len = self.roi_shuffle_index(x,v2d_roi_len, mask_ratio, N, 'v2d')
        v2v_shuffle_index, v2v_keep_len = self.roi_shuffle_index(x,v2v_roi_len, mask_ratio, N, 'v2v')
        v3d_shuffle_index, v3d_keep_len = self.roi_shuffle_index(x,v3d_roi_len, mask_ratio, N, 'v3d')
        v3v_shuffle_index, v3v_keep_len = self.roi_shuffle_index(x,v3v_roi_len, mask_ratio, N, 'v3v')
        v4_shuffle_index, v4_keep_len = self.roi_shuffle_index(x,v4_roi_len, mask_ratio, N, 'v4')
        
        total_shuffle_index = torch.cat([v1d_shuffle_index, v1v_shuffle_index, v2d_shuffle_index, v2v_shuffle_index, 
                                         v3d_shuffle_index, v3v_shuffle_index, v4_shuffle_index], dim=-1)


        # sort noise for each sample
        ids_shuffle = torch.argsort(total_shuffle_index, dim=1)  # ascend: small is keep, large is remove 
        ids_restore = torch.argsort(ids_shuffle, dim=1) # 재배치 된 배열에서 다시 원래 배열로 돌리기 위한 index

        # keep the first subset
        len_keep = v1d_keep_len + v1v_keep_len + v2d_keep_len + v2v_keep_len + v3d_keep_len + v3v_keep_len + v4_keep_len
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # B, non_maskedP, embed

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:,:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # before shuffle, masked index information

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # x is non-masked(0) patch
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        #x = torch.cat((cls_tokens, x), dim=1) # B, 1(cls)+non_maskedP, embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # N 
        #x = self.avgpool(x.transpose(1, 2))  # B L 1
        #x = torch.flatten(x, 1)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x) # B, non_maskedP, d_embed

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # B, maskedP, d_embed
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1], 1) # B, maskedP, d_embed
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token  111111100000000000
        # B, non_maskedP + maskedP, d_embed
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle, patch location
        # x_: B, nP, d_embed -> original location
        #x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x_ + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x) # B, nP, patch_size -> each patch predict patchify value(16), not predict patch embedding

        # remove cls token
        #x = x[:, 1:, :]

        return x # B, nP, patch_size

          
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, num_voxels]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs) # B, num_patches, patch_size

        loss = (pred - target) ** 2
        #loss = torch.abs(pred - target)
        loss = loss.mean(dim=-1)  # [B, nP], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches, mask -> 1 keep ->0
        return loss

    def forward(self, imgs, valid=0): #imgs {roi: [B, voxel_index]}

      
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio) # B, 13*7, 1024
        B, N, D = latent.size()
        
        if valid>0:
            latent_mask = nn.Parameter(torch.zeros(B, self.encoder_window_size, self.embed_dim))
            if valid > 1:
                latent[:,:(valid-1)*self.encoder_window_size,:] = latent_mask.repeat(1,(valid-1),1)
            if valid < 7:    
                latent[:,valid*self.encoder_window_size:,:] = latent_mask.repeat(1,(7-valid),1)
            
            #latent[:,(valid-1)*self.encoder_window_size:valid*self.encoder_window_size,:] = latent_mask
        
        pred = self.forward_decoder(latent, ids_restore)  # [B, nP, p]
        loss = self.forward_loss(imgs, pred, mask) # !!!!!!

        return loss, pred, mask



class fmri_encoder(nn.Module):
    def __init__(self, num_voxels=53*16*7, patch_size=16, embed_dim=1024, window_size=53, in_chans=1,
                 depth=24, num_heads=16, mlp_ratio=1., norm_layer=nn.LayerNorm, global_pool=False):
        super().__init__()
        self.patch_embed = PatchEmbed1D(num_voxels, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            BasicLayer(dim=embed_dim, input_resolution=num_patches, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                      qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,  norm_layer=norm_layer,
                      use_checkpoint=False,fused_window_process=False,lobe_wise=False, decoder_tf=False)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # not mask
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)
        x = self.norm(x)

        return x  

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)  # N, n_seq, embed_dim
        latent = self.forward_encoder(imgs) # N, n_seq, embed_dim
        return latent # N, n_seq, embed_dim
    
    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 
