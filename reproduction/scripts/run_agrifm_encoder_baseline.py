from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
AGRIFM_ROOT = ROOT / "AgriFM-main"


class _CompatRegistry:
    """Minimal MMEngine Registry subset used by the released AgriFM encoder."""

    def __init__(self, name: str):
        self.name = name
        self._modules: dict[str, type] = {}

    def register_module(self, name: str | None = None):
        def decorator(cls):
            self._modules[name or cls.__name__] = cls
            return cls

        return decorator

    def build(self, config: dict):
        values = dict(config)
        module_type = values.pop("type")
        if isinstance(module_type, str):
            if module_type not in self._modules:
                raise KeyError(f"{module_type!r} is not registered in {self.name}")
            module_type = self._modules[module_type]
        return module_type(**values)


class _CompatBaseModel(torch.nn.Module):
    def __init__(self, data_preprocessor=None, init_cfg=None):
        super().__init__()
        self.data_preprocessor = data_preprocessor
        self.init_cfg = init_cfg


def _compat_load_checkpoint(
    module: torch.nn.Module, checkpoint: str, strict: bool = False,
    revise_keys=None, **kwargs,
):
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if revise_keys:
        import re

        state = {
            _apply_revisions(key, revise_keys, re): value
            for key, value in state.items()
        }
    module.load_state_dict(state, strict=strict)
    return state


def _apply_revisions(key: str, revise_keys, re_module) -> str:
    for pattern, replacement in revise_keys:
        key = re_module.sub(pattern, replacement, key)
    return key


def _install_mmengine_compat() -> dict[str, types.ModuleType | None]:
    names = (
        "mmengine", "mmengine.registry", "mmengine.model", "mmengine.runner",
    )
    saved = {name: sys.modules.get(name) for name in names}
    mmengine = types.ModuleType("mmengine")
    registry = types.ModuleType("mmengine.registry")
    model = types.ModuleType("mmengine.model")
    runner = types.ModuleType("mmengine.runner")
    registry.Registry = _CompatRegistry
    model.BaseModule = _CompatBaseModel
    model.BaseModel = _CompatBaseModel
    runner.load_checkpoint = _compat_load_checkpoint
    sys.modules["mmengine"] = mmengine
    sys.modules["mmengine.registry"] = registry
    sys.modules["mmengine.model"] = model
    sys.modules["mmengine.runner"] = runner
    return saved


class _CompatDropPath(torch.nn.Module):
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
        random_tensor.floor_()
        return values.div(keep_prob) * random_tensor


def _install_timm_compat() -> dict[str, types.ModuleType | None]:
    names = ("timm", "timm.models", "timm.models.layers")
    saved = {name: sys.modules.get(name) for name in names}
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    layers.DropPath = _CompatDropPath
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.layers"] = layers
    return saved


def _compat_rearrange(values: torch.Tensor, pattern: str, **axes) -> torch.Tensor:
    normalized = " ".join(pattern.split())
    if normalized in (
        "b c d h w -> b d h w c",
        "n c d h w -> n d h w c",
    ):
        return values.permute(0, 2, 3, 4, 1)
    if normalized in (
        "b d h w c -> b c d h w",
        "n d h w c -> n c d h w",
    ):
        return values.permute(0, 4, 1, 2, 3)
    if normalized == "b c t h w -> (b c) 1 t h w":
        b, c, t, h, w = values.shape
        return values.reshape(b * c, 1, t, h, w)
    if normalized == "(b c) d t h w -> b c d t h w":
        b, c = int(axes["b"]), int(axes["c"])
        _, d, t, h, w = values.shape
        return values.reshape(b, c, d, t, h, w)
    if normalized == "b c d t h w -> (b c) d t h w":
        b, c, d, t, h, w = values.shape
        return values.reshape(b * c, d, t, h, w)
    if normalized == "(b c) 1 t h w -> b c 1 t h w":
        b, c = int(axes["b"]), int(axes["c"])
        _, one, t, h, w = values.shape
        return values.reshape(b, c, one, t, h, w)
    raise NotImplementedError(f"Unsupported AgriFM rearrange pattern: {pattern}")


def _compat_einops_reduce(
    values: torch.Tensor, pattern: str, reduction: str, **axes,
) -> torch.Tensor:
    normalized = " ".join(pattern.split())
    if normalized != "b c (t s) h w -> b c t h w" or reduction != "mean":
        raise NotImplementedError(
            f"Unsupported AgriFM reduce pattern: {pattern!r}, {reduction!r}"
        )
    b, c, combined, h, w = values.shape
    scale = int(axes["s"])
    if combined % scale:
        raise ValueError("Temporal dimension is not divisible by reduction scale")
    return values.reshape(b, c, combined // scale, scale, h, w).mean(dim=3)


def _install_einops_compat() -> dict[str, types.ModuleType | None]:
    saved = {"einops": sys.modules.get("einops")}
    module = types.ModuleType("einops")
    module.rearrange = _compat_rearrange
    module.reduce = _compat_einops_reduce
    sys.modules["einops"] = module
    return saved


def _encoder_config() -> dict:
    return {
        "type": "PretrainingSwinTransformer3DEncoder",
        "patch_emd_cfg": {
            "type": "SwinPatchEmbed3D",
            # The released checkpoint tensor is [128, 10, 4, 4, 4].
            # This is authoritative over the stale (4, 2, 2) example config.
            "patch_size": (4, 4, 4),
            "in_chans": 10,
            "embed_dim": 128,
        },
        "backbone_cfg": {
            "type": "SwinTransformer3D",
            "pretrained": None,
            "pretrained2d": False,
            "patch_size": (4, 2, 2),
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32],
            "window_size": (8, 7, 7),
            "out_indices": (0, 1, 2, 3),
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "qk_scale": None,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.2,
            "patch_norm": False,
            "frozen_stages": -1,
            "use_checkpoint": False,
            "downsample_steps": ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
            "feature_fusion": "cat",
            "mean_frame_down": True,
        },
    }


def _remap_checkpoint(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped = {}
    for key, value in state_dict.items():
        if key == "encoder.S2_patch_emd.weights":
            mapped["patch_emd.proj.weight"] = value
        elif key == "encoder.S2_patch_emd.bias":
            mapped["patch_emd.proj.bias"] = value
        elif key.startswith("encoder.backbone."):
            mapped[key.removeprefix("encoder.")] = value
    return mapped


def _load_video_swin_module():
    """Load only the encoder source without importing optional mmcv C++ ops."""
    compat_saved = None
    try:
        from mmengine.registry import Registry
    except ModuleNotFoundError:
        compat_saved = _install_mmengine_compat()
        from mmengine.registry import Registry
    timm_saved = None
    try:
        from timm.models.layers import DropPath as _TimmDropPath  # noqa: F401
    except ModuleNotFoundError:
        timm_saved = _install_timm_compat()
    einops_saved = None
    try:
        from einops import rearrange as _einops_rearrange  # noqa: F401
    except ModuleNotFoundError:
        einops_saved = _install_einops_compat()

    models = Registry("agrifm_models")
    transforms = Registry("agrifm_transforms")
    module_names = ("mmseg", "mmseg.models", "mmseg.registry")
    saved = {name: sys.modules.get(name) for name in module_names}
    saved_leaf = {
        name: sys.modules.get(name)
        for name in ("mmseg.models.builder", "mmseg.registry.registry")
    }
    try:
        for name in module_names:
            sys.modules[name] = types.ModuleType(name)
        builder = types.ModuleType("mmseg.models.builder")
        builder.BACKBONES = models
        registry = types.ModuleType("mmseg.registry.registry")
        registry.MODELS = models
        registry.TRANSFORMS = transforms
        sys.modules[builder.__name__] = builder
        sys.modules[registry.__name__] = registry

        source = AGRIFM_ROOT / "AgriFM" / "models" / "video_swin_transformer.py"
        spec = importlib.util.spec_from_file_location("_physcrop_agrifm_video_swin", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, models
    finally:
        for name, value in {**saved, **saved_leaf}.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        if compat_saved is not None:
            for name, value in compat_saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        if timm_saved is not None:
            for name, value in timm_saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        if einops_saved is not None:
            for name, value in einops_saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


def build_encoder(checkpoint: str, device: str = "cpu"):
    _, models = _load_video_swin_module()
    encoder = models.build(_encoder_config())
    raw_state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(raw_state, dict):
        raise TypeError(f"Expected a state dict in {checkpoint}, got {type(raw_state)!r}")
    state = _remap_checkpoint(raw_state)
    missing, unexpected = encoder.load_state_dict(state, strict=False)

    ignored_missing = {key for key in missing if key.endswith("relative_position_index")}
    real_missing = sorted(set(missing) - ignored_missing)
    if real_missing or unexpected:
        raise RuntimeError(
            "AgriFM checkpoint mapping is incomplete: "
            f"missing={real_missing[:10]}, unexpected={unexpected[:10]}"
        )
    encoder.eval().to(device)
    return encoder


__all__ = ["build_encoder"]
