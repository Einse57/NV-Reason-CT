# SPDX-License-Identifier: OpenMDW-1.1
"""Shared helpers for the NV-Reason-CT 3D encoder OpenVINO tooling.

The reference encoder is the upstream ``Vision3D`` class from the Hugging Face
model snapshot (``model.py``), loaded in float32 on CPU. Only the
``model.vision3d.*`` tensors are read from ``model.safetensors``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

INPUT_SHAPE = (1, 1, 192, 192, 192)
PREFIX = "model.vision3d."


def _import_from(model_dir: Path, name: str):
    path = model_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"nvreason_{name}", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_vision3d_state(model_dir: Path, weights: Path | None = None):
    """Return the vision3d state dict (keys without the ``model.vision3d.`` prefix)."""
    from safetensors import safe_open

    weights = Path(weights) if weights else model_dir / "model.safetensors"
    state = {}
    with safe_open(str(weights), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not a dict
            if key.startswith(PREFIX):
                state[key[len(PREFIX) :]] = f.get_tensor(key)
            elif key.startswith(("sub_vision.", "merger.")):
                state[key] = f.get_tensor(key)
    if not state:
        raise RuntimeError(f"no vision3d tensors found in {weights.name}")
    return state


def build_reference_encoder(model_dir: str | Path, weights: str | Path | None = None):
    """Build upstream ``Vision3D`` in float32 eval mode with checkpoint weights."""
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    hf_model = _import_from(model_dir, "model")
    vision_cfg = Qwen3_5VisionConfig(**cfg["vision_config"])
    encoder = hf_model.Vision3D(
        vision_cfg,
        input_shape=tuple(cfg.get("vit3d_input_shape", INPUT_SHAPE[2:])),
        patch_embed_size=tuple(cfg.get("vit3d_patch_embed_size", (8, 8, 8))),
    )
    state = load_vision3d_state(model_dir, weights)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"vision3d weight mismatch: missing={missing} unexpected={unexpected}"
        )
    return encoder.float().eval().requires_grad_(False), torch


def load_input(
    spec: str, model_dir: str | Path | None = None, region: str = "chest"
) -> np.ndarray:
    """Load an encoder input of shape [1, 1, 192, 192, 192] (float32).

    ``spec`` is ``synthetic[:seed]`` (uniform in [-1, 1], the post-normalization
    range), a ``.npy`` file, or a NIfTI CT volume preprocessed with the upstream
    ``ImageLoader3D`` (CT normalization, LPS, 2 mm, anatomy crop ``region``).
    """
    if spec.startswith("synthetic"):
        seed = int(spec.split(":", 1)[1]) if ":" in spec else 0
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, size=INPUT_SHAPE).astype(np.float32)
    if spec.endswith(".npy"):
        arr = np.load(spec).astype(np.float32)
    else:
        if model_dir is None:
            raise ValueError("model_dir is required to preprocess NIfTI input")
        processor = _import_from(Path(model_dir), "processor")
        loader = processor.ImageLoader3D()
        arr = loader.load_image(
            spec, normalize_mode="ct", anatomy_region=region
        ).numpy()
        arr = arr.astype(np.float32)
    arr = arr.reshape(INPUT_SHAPE)
    return arr


def compare(ref: np.ndarray, out: np.ndarray) -> dict:
    """Agreement metrics between a reference and a candidate token matrix."""
    a = ref.astype(np.float64).reshape(-1, ref.shape[-1])
    b = out.astype(np.float64).reshape(-1, out.shape[-1])
    diff = np.abs(a - b)
    tok_cos = (a * b).sum(1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    )
    return {
        "cosine": float(
            (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        ),
        "token_cosine_min": float(tok_cos.min()),
        "token_cosine_mean": float(tok_cos.mean()),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_l2": float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)),
        "ref_abs_max": float(np.abs(a).max()),
    }


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (Windows peak working set)."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak:
            return peak / 2**20
    except ImportError:
        pass
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024 if sys.platform != "darwin" else ru / 2**20
