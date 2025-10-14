import torch
import torch.nn as nn
import torch.nn.functional as F # This gives us the softmax()
import math
from typing import Optional
import timm
import torch
import torch.nn as nn


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    Uses conv2d for patch extraction.
    The input is an image of shape (batch_size, num_channels, image_size, image_size)
    The output is a tensor of shape (batch_size, num_patches, hidden_size
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class Embeddings(nn.Module):
    """
    Here we add cls token and positional embeddings
    These are learnable positional embeddings that are added to the patch embeddings.
    """
    def __init__(self, config):
       super().__init__()
       self.config =  config
       self.patcher = PatchEmbeddings(self.config)
       self.cls_tokens = nn.Parameter((torch.randn(1,1, config['hidden_size'])))
       self.pos_embs = nn.Parameter((torch.randn(1,self.patcher.num_patches+1, config['hidden_size'])))

    def forward(self, x):
        x = self.patcher(x)
        bs = x.size()[0]
        print(f"this is bs: {bs}")
        # add cls token
        cls_token =  self.cls_tokens.expand(bs, -1, -1)
        x = torch.cat((x,cls_token), dim = 1)
        # add position embeddings 
        x = x + self.pos_embs
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)
    

class Backbone(nn.Module):
    """
    The backbone of the model.
    This module contains the patch embeddings, positional embeddings, and multi-head attention.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.attention_heads = nn.ModuleList([
            AttentionHead(config['hidden_size'], config['attention_head_size'], config['dropout'])
            for _ in range(config['num_attention_heads'])
        ])
    
    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches+1, hidden_size)
        x = self.embeddings(x)
        # Apply each attention head
        for head in self.attention_heads:
            x, _ = head(x)
        return x

# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------





class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
        )

        pixel_mean = torch.tensor(self.backbone.default_cfg["mean"]).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(self.backbone.default_cfg["std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)