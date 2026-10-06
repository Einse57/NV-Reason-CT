#!/usr/bin/env python3
"""Minimal Primus+merger export — avoids loading full transformers VLM stack."""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "hf_src"
OUT = ROOT / "ir" / "encoder_upstream"
OUT.mkdir(parents=True, exist_ok=True)

from dynamic_network_architectures.architectures.primus import Primus
from safetensors import safe_open


class PatchMerger(nn.Module):
    """Qwen3.5VisionPatchMerger without importing transformers modeling."""

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 1):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = None  # filled below to match checkpoint naming
        # Checkpoint uses: merger.norm.{weight,bias}, merger.linear_fc1/2
        self.norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear_fc1(x)
        x = self.act(x)
        x = self.linear_fc2(x)
        return x


class Vision3DMinimal(nn.Module):
    def __init__(self, out_hidden_size=2560, input_shape=(192, 192, 192), patch_embed_size=(8, 8, 8)):
        super().__init__()
        self.sub_vision = Primus(
            input_channels=1,
            num_classes=1,
            eva_depth=16,
            eva_numheads=12,
            embed_dim=864,
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            use_rot_pos_emb=True,
            use_abs_pos_embed=False,
            drop_path_rate=0.2,
            init_values=0.1,
            scale_attn_inner=True,
            num_register_tokens=0,
        )
        self.sub_vision.up_projection = nn.Identity()
        primus_embed_dim = self.sub_vision.eva.embed_dim
        self.merger = PatchMerger(dim=out_hidden_size, context_dim=primus_embed_dim, spatial_merge_size=1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        x = self.sub_vision(volume)  # [B, 864, T, H, W]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return self.merger(x.view(-1, x.shape[-1]))


def main():
    cfg = json.loads((HF / "config.json").read_text())
    out_h = cfg["vision_config"]["out_hidden_size"]
    input_shape = tuple(cfg.get("vit3d_input_shape", [192, 192, 192]))
    patch = tuple(cfg.get("vit3d_patch_embed_size", [8, 8, 8]))
    print("build", input_shape, patch, out_h, flush=True)
    model = Vision3DMinimal(out_hidden_size=out_h, input_shape=input_shape, patch_embed_size=patch).eval()

    sd = {}
    with safe_open(str(HF / "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("model.vision3d."):
                sd[k[len("model.vision3d.") :]] = f.get_tensor(k)
    # Checkpoint may use GELU naming inside merger differently; try load
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing", len(missing), missing[:20], flush=True)
    print("unexpected", len(unexpected), unexpected[:20], flush=True)
    if any("merger" in m for m in missing):
        # Inspect actual merger key names in ckpt
        print("merger ckpt keys", [k for k in sd if k.startswith("merger")][:20])
        print("merger model keys", list(model.merger.state_dict().keys()))
        sys.exit(2)

    del sd
    gc.collect()
    model = model.float()
    dummy = torch.zeros(1, 1, *input_shape, dtype=torch.float32)
    with torch.inference_mode():
        out_torch = model(dummy).detach().cpu().numpy()
    print("torch", out_torch.shape, float(out_torch.mean()), float(out_torch.std()), flush=True)
    np.save(OUT / "torch_zero.npy", out_torch)

    onnx_path = OUT / "vision_fp32.onnx"
    print("onnx export...", flush=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["volume"],
        output_names=["tokens"],
        opset_version=17,
        dynamo=False,
    )
    print("onnx MB", onnx_path.stat().st_size / 1e6, flush=True)
    del model, dummy
    gc.collect()

    import openvino as ov

    print("ov convert...", flush=True)
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(OUT / "vision.xml"))
    print("IR MB", (OUT / "vision.bin").stat().st_size / 1e6, flush=True)
    # drop onnx to save disk
    onnx_path.unlink(missing_ok=True)
    for p in OUT.glob("vision_fp32*"):
        if p.suffix in {".onnx", ".data"} or "vision_fp32" in p.name:
            try:
                p.unlink()
            except Exception:
                pass

    core = ov.Core()
    compiled = core.compile_model(str(OUT / "vision.xml"), "CPU")
    req = compiled.create_infer_request()
    arr = np.zeros((1, 1, *input_shape), np.float32)
    req.infer({0: arr})
    out_ov = req.get_output_tensor(0).data
    a = out_torch.astype(np.float64).ravel()
    b = out_ov.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"cosine={cos:.8f} max_abs={float(np.max(np.abs(a - b))):.6e}", flush=True)
    np.savez_compressed(OUT / "zero_volume_ref.npz", torch=out_torch, ov=out_ov, cosine=np.array([cos]))
    print("ENCODER_UPSTREAM_OK", flush=True)


if __name__ == "__main__":
    main()
