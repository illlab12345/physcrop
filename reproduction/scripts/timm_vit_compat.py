"""Small timm Vision Transformer compatibility layer for frozen SatMAE inference.

This implements only the timm modules used by models_vit_group_channels.py. It
preserves the parameter names and forward equations required by the released
SatMAE checkpoint; it is not a general timm replacement.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable

import torch
from torch import nn


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return values
        keep_prob = 1.0 - self.drop_prob
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=values.dtype, device=values.device
        )
        return values.div(keep_prob) * random_tensor.floor_()


class PatchEmbed(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
        norm_layer: Callable | None = None, flatten: bool = True, **kwargs,
    ):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.proj(values)
        if self.flatten:
            values = values.flatten(2).transpose(1, 2)
        return self.norm(values)


class Mlp(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int | None = None,
        out_features: int | None = None, act_layer: Callable = nn.GELU,
        drop: float = 0.0, **kwargs,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.drop1(self.act(self.fc1(values)))
        return self.drop2(self.fc2(values))


class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
        qk_norm: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0,
        norm_layer: Callable = nn.LayerNorm, **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = values.shape
        qkv = self.qkv(values).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = self.attn_drop(attention.softmax(dim=-1))
        values = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(values))


class Block(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
        qkv_bias: bool = False, qk_norm: bool = False, drop: float = 0.0,
        attn_drop: float = 0.0, drop_path: float = 0.0,
        act_layer: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.drop_path1(self.attn(self.norm1(values)))
        return values + self.drop_path1(self.mlp(self.norm2(values)))


class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        qkv_bias=True, qk_norm=False, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.0, norm_layer: Callable = nn.LayerNorm,
        act_layer: Callable = nn.GELU, **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_norm=qk_norm, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_paths[index],
                norm_layer=norm_layer, act_layer=act_layer,
            )
            for index in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, values: torch.Tensor) -> torch.Tensor:
        values = self.patch_embed(values)
        cls = self.cls_token.expand(values.shape[0], -1, -1)
        values = self.pos_drop(torch.cat((cls, values), dim=1) + self.pos_embed)
        values = self.blocks(values)
        return self.norm(values)[:, 0]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(values))


def install() -> None:
    if "timm.models.vision_transformer" in sys.modules:
        return
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    vision = types.ModuleType("timm.models.vision_transformer")
    layers = types.ModuleType("timm.models.layers")
    vision.PatchEmbed = PatchEmbed
    vision.Attention = Attention
    vision.Mlp = Mlp
    vision.Block = Block
    vision.VisionTransformer = VisionTransformer
    layers.DropPath = DropPath
    layers.trunc_normal_ = nn.init.trunc_normal_
    timm.models = models
    models.vision_transformer = vision
    models.layers = layers
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.vision_transformer"] = vision
    sys.modules["timm.models.layers"] = layers
