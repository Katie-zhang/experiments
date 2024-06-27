import torch
import torch.nn as nn
import torch.nn.functional as F
from models.patchembed import PatchEmbed23D
import timm

class FeedForward(nn.Module):
    def __init__(self,embed_dim, hidden_dim, dropout_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        return self.net(x) # [B, N, E]
    
    
    
class Attention(nn.Module):
    def __init__(self, embed_dim, number_of_heads, dropout_rate=0.0):
        super().__init__()

        assert embed_dim % number_of_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = number_of_heads
        self.head_dim = embed_dim // number_of_heads
        self.scale = self.head_dim ** -0.5 # scale factor
        

        self.to_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        
        self.att_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout_rate)

        
    
    def forward(self, x):
        B, N, C = x.shape

        qkv = self.to_qkv(x) # [B, N, 3*E]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim) # [B, N, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, num_heads, N, head_dim]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # [B, num_heads, N, N]
        attn = F.softmax(attn, dim=-1) # get the attention weights [B, num_heads, N, N]

        x = (attn @ v) # [B, num_heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x) 
        x = self.proj_drop(x)

        return x
        


class Transformer(nn.Module):
    def __init__(self, embed_dim, number_of_heads, mlp_hidden_dim, dropout_rate=0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = Attention(embed_dim, number_of_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, mlp_hidden_dim, dropout_rate)
        
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x)) # [B, N, E]
        x = x + self.ff(self.norm2(x)) # [B, N, E]
        return x
    
    
class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, num_classes_list):
        super(MultiTaskHead, self).__init__()
        self.task_heads = nn.ModuleList([
            nn.Linear(input_dim, num_classes) for num_classes in num_classes_list
        ])

    def forward(self, x, task_id):
        task_output = self.task_heads[task_id](x)
        return task_output # [B, num_classes]


class ViT(nn.Module):
    def __init__(self,  patch_size, num_classes_list, embed_dim):
        super().__init__()
        self.patch_embed = PatchEmbed23D(img_size=224, patch_size=16, in_chans=3, embed_dim=embed_dim) 
        num_patches2 = self.patch_embed.num_patches2
        num_patches3 = self.patch_embed.num_patches3

        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches2 + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed3 = nn.Parameter(torch.zeros(1, num_patches3 + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
       # self.transformer = nn.Sequential(*[Transformer(embed_dim, number_of_heads, mlp_hidden_dim, dropout_rate) for _ in range(num_layers)])
       
        # substitute the transformer layer
       # self.base_model = timm.create_model('vit_small_patch16_224', pretrained=True)
       # self.base_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
        self.base_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
        
        self.transformer = self.base_model.blocks
        self.norm = nn.LayerNorm(embed_dim)
        self.multi_head = MultiTaskHead(embed_dim, num_classes_list)
    
    def forward(self, x, task_id):
        #print("Input shape:", x.shape)

       # B, C, H, W = x.shape
        B = x.shape[0]
        x,t= self.patch_embed(x) # [B, N, C]
        
        if t == 2:
            pos_embed = self.pos_embed2
        else:
            pos_embed = self.pos_embed3
        

        x = x + pos_embed[:, 1:, :]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) # [B, N+1, C]
        
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, 0] 
   
      
        # Multi head
        x = self.multi_head(x, task_id)
        
        return x    