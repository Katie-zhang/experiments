import torch
import torch.nn as nn
from timm.models.layers import to_2tuple,to_3tuple



    

class PatchEmbed23D(nn.Module):
    """ 2D Image to Patch Embedding
        This implementation is based on the code from the paper:
        "Uni4Eye: Unified 2D and 3D Self-supervised Pre-training via Masked Image Modeling Transformer for Ophthalmic Image Classification"
        by Zhiyuan Cai(2022)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, norm_layer=None, flatten=True):
        super().__init__()
        img_size2 = to_2tuple(img_size)
        img_size3 = (img_size, img_size, img_size//2)
        patch_size2 = to_2tuple(patch_size)
        patch_size3 = to_3tuple(patch_size)
        self.img_size2 = img_size2
        self.img_size3 = img_size3
        self.patch_size2 = patch_size2
        self.patch_size3 = patch_size3
        self.grid_size2 = (img_size2[0] // patch_size2[0], img_size2[1] // patch_size2[1])
        self.grid_size3 = (img_size3[0] // patch_size3[0], img_size3[1] // patch_size3[1], img_size3[2] // patch_size3[2])
        self.num_patches2 = self.grid_size2[0] * self.grid_size2[1]
        self.num_patches3 = self.grid_size3[0] * self.grid_size3[1] * self.grid_size3[2]
        self.flatten = flatten

        self.proj2= nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size2, stride=patch_size2)
        self.proj3= nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size3, stride=patch_size3)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = self.proj2(x)
            t = 2
        else:
            B, C, H, W, D = x.shape
            x = self.proj3(x)
            t = 3
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, t
    
    
    
    
# class PatchEmbed2d(nn.Module):
#     def __init__(self, in_channels, patch_size, emb_size):
#         super().__init__()
#         self.patch_size = patch_size
#         self.emb_size = emb_size
#         self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
    
#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         x = self.projection(x) #[B, emb_size, H//patch_size, W//patch_size]
#         x = x.flatten(2) #[B, emb_size, H//patch_size * W//patch_size]
#         x = x.transpose(1, 2) #[B, H//patch_size * W//patch_size, emb_size]
#         return x
    
    
# class PatchEmbed3d(nn.Module):
#     def __init__(self, in_channels, patch_size, emb_size, num_patches):
#         super().__init__()
#         self.patch_size = patch_size
#         self.emb_size = emb_size
#         self.num_patches = num_patches  #
#         self.projection = nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
#         self.pooling = nn.AdaptiveAvgPool3d((num_patches, 1, 1))  #

#     def forward(self, x):
#         B, C, D, H, W = x.shape
#         x = self.projection(x) # [B, emb_size, D//patch_size, H//patch_size, W//patch_size]
#         x = self.pooling(x)  # [B, emb_size, num_patches, 1, 1]
#         x = x.flatten(2)
#         x = x.transpose(1, 2)
#         return x