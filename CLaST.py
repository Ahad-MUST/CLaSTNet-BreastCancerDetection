import torch
import torch.nn as nn
from Blocks import ECBlock, PatchEmbed, ViTBlock, PatchMerging

import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

class CLaST(nn.Module):
    """
    CLaST: A hybrid model combining convolutional and transformer-based layers for image classification.
    """

    def __init__(self, img_size, patch_size, num_classes,
                 embed_dim, conv_depths, depths, num_heads,
                 window_size, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate,
                 norm_layer, patch_norm, multi_shift,
                 shift_window):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.num_conv_layers = conv_depths
        self.num_features = int(embed_dim * 2 ** (self.num_layers + self.num_conv_layers))

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # EClock Layers (Convolutional Blocks for feature extraction)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(ECBlock(int(embed_dim), int(embed_dim * 2), stride=2, expansion=1))
        self.conv_layers.append(ECBlock(int(embed_dim * 2), int(embed_dim * 2 ** 2), stride=2, expansion=2))
        self.conv_layers.append(ECBlock(int(embed_dim * 2 ** self.num_conv_layers), int(embed_dim * 2 ** (self.num_conv_layers+1)), stride=2))
    
        # Transformer-based layers for capturing global dependencies
        self.layers = nn.ModuleList([
            ViTBlock(
                dim=int(embed_dim * 2 ** (self.num_conv_layers+1)),
                input_resolution=(patches_resolution[0] // (2 ** (self.num_conv_layers+1)),
                                  patches_resolution[1] // (2 ** (self.num_conv_layers+1))),
                depth=depths[0], num_heads=num_heads, window_size=window_size, mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                norm_layer=norm_layer, downsample=PatchMerging,
                shift_window=shift_window, multi_shift=multi_shift
            )
        ])

        # Normalization and Classification Head
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
  
    def forward(self, x):
        """
        Forward pass through CLaST model.
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
        Returns:
            torch.Tensor: Class logits
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Convolutional feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten and transpose for transformer layers
        x = x.flatten(2).transpose(1, 2)
        
        # Transformer-based processing
        for layer in self.layers:
            x = layer(x)      
   
        # Final normalization and classification
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x
