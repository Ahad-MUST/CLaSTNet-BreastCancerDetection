import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.layers import DropPath, trunc_normal_

import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

class ECBlock(nn.Module):
    """
    Efficient Convolutional Block (ECBlock):
    - Expands the input channels using pointwise convolution
    - Applies depthwise convolution for spatial processing
    - Reduces back to output channels
    """
    def __init__(self, inp, oup, stride, expansion=2):
        super().__init__()
        self.stride = stride
        expansion_dim = inp * expansion

        self.conv = nn.Sequential(
            # Pointwise Convolution: Expands the channels
            nn.Conv2d(inp, expansion_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expansion_dim),
            nn.ReLU6(),  # ReLU6 activation function

            # Depthwise Convolution: Applies spatial filtering
            nn.Conv2d(expansion_dim, oup, kernel_size=3, stride=stride, padding=1, groups=expansion_dim, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(),  # Activation function
        )

    def forward(self, x):
        return self.conv(x)

class PatchEmbed(nn.Module):
    """
    Patch Embedding Layer:
    - Converts an input image into a sequence of patches with embedded features.
    """
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        
        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.embed_dim = embed_dim

        # Convolutional projection for patch embedding
        self.proj = nn.Sequential(
            # First convolution: Reducing spatial resolution
            nn.Conv2d(3, self.embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU6(),
            nn.BatchNorm2d(self.embed_dim // 2),
            
            # Second convolution: Expanding to embedding dimension
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU6(),
            nn.BatchNorm2d(self.embed_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        return self.proj(x)


def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., k_conv_value=1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.k_conv_value = k_conv_value
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, in_v=None, in_kv=None):

        B_, N, C = x.shape
        H, W = self.window_size
        

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # B_, head, N, d
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)


        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class LadderSABlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, shift_direction=1,
                 k_conv_value=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.shift_direction = shift_direction
        
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        pool_size =1
        self.pool = nn.AvgPool2d(pool_size, 1, pool_size//2)
        block= WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            k_conv_value=k_conv_value)
        self.attn = block
        #   print(block)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        attn_mask = None 

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, in_v=None):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)

        shifted_x = x        

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask, in_v=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C   # H, W

        x = shifted_x
        
        # print(x.shape)
        x = x.view(B, H*W, C)

        x = shortcut + self.drop_path(x)

        return x


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):

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

class multi_branch_LadderBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, branch2input_conv=False,
                 multi_shift=False,
                 kv_down_scale=0,):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_branch = len(window_size)
        self.blocks = []
        self.conv_before_branch = None
        dim = int(dim / self.num_branch)
        self.branch2input_conv = branch2input_conv
        if branch2input_conv:
            self.branch_conv = []

        shift_direction = 1

        for i in range(self.num_branch):
            if multi_shift:
                if i == 0 or kv_down_scale > 0:
                    shift_size = 0
                elif i == 1:
                    shift_size = window_size[i] // 2
                    shift_direction = 1  # not assigned in multi_shift
                elif i == 2:
                    shift_size = window_size[i] // 2
                    shift_direction = 0
                else:
                    print('not implement for more than 3 branches!!')
                    exit()
            
            else:
                shift_size = window_size[i] // 2 if shift_size == None else shift_size
            
            block = LadderSABlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, shift_size=shift_size, mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path, norm_layer=norm_layer, 
                                shift_direction=shift_direction)
            
            #print(block)
            self.blocks.append(block)
        
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"
        dim = int(C / self.num_branch)

        for i in range(self.num_branch):

            x_input = x[:,:,dim*i:dim*(i+1)]
            in_v = None
            
            x_ = self.blocks[i](x_input, in_v)
            xs = torch.cat((xs, x_), dim=2) if i>0 else x_
            
        shortcut = xs         
        
        xs = xs + shortcut
        return xs

class ViTBlock(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 branch2input_conv=False, aggressive_dim=False, shift_window=True,
                 multi_shift=False, kv_down_scale=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.aggressive_dim = aggressive_dim

        # build blocks
        self.blocks = []

        self.blocks = nn.ModuleList(self.blocks)
        self.blocks = nn.ModuleList([
            multi_branch_LadderBlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, window_size=window_size,
                                shift_size=0 if (i % 2 == 0) or not shift_window else None,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale, kv_down_scale=kv_down_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer, branch2input_conv=branch2input_conv, multi_shift=multi_shift)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x