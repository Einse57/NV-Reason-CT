# SPDX-License-Identifier: OpenMDW-1.1
"""Check the NV-Reason-CT 3D encoder on a device against a float32 CPU reference.

Runs only the vision tower (Primus + merger) on one preprocessed CT crop, first
in float32 on the CPU and then on the requested device and dtype, and prints
agreement metrics and timing. Exits with status 1 if the cosine similarity is
below --min-cosine.

Example:
    python scripts/check_encoder_device.py examples/example_1.nii.gz \
        --region chest --device xpu --dtype float16
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

DEFAULT_MODEL = "nvidia/NV-Reason-CT"
PREFIX = "model.vision3d."
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ct_path", help="Path to a .nii or .nii.gz CT volume")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--region", choices=("chest", "abdomen"), default="chest")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "xpu", "cpu"),
        default="auto",
        help="Device to check (default: auto, the first available of cuda, xpu, cpu)",
    )
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs on the device")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    return parser.parse_args()


def select_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def model_file(model, name):
    local = Path(model) / name
    if local.is_file():
        return str(local)
    return hf_hub_download(model, name)


def load_encoder(model):
    """Build the upstream Vision3D tower in float32 with its checkpoint weights."""
    cfg = json.loads(Path(model_file(model, "config.json")).read_text())
    vision3d_cls = get_class_from_dynamic_module("model.Vision3D", model)
    encoder = vision3d_cls(
        Qwen3_5VisionConfig(**cfg["vision_config"]),
        input_shape=tuple(cfg.get("vit3d_input_shape", (192, 192, 192))),
        patch_embed_size=tuple(cfg.get("vit3d_patch_embed_size", (8, 8, 8))),
    )
    try:
        index = json.loads(
            Path(model_file(model, "model.safetensors.index.json")).read_text()
        )
        shards = sorted(
            {f for k, f in index["weight_map"].items() if k.startswith(PREFIX)}
        )
    except Exception:
        shards = ["model.safetensors"]
    state = {}
    for shard in shards:
        with safe_open(model_file(model, shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(PREFIX):
                    state[key[len(PREFIX) :]] = f.get_tensor(key)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"vision3d weights: missing={missing} unexpected={unexpected}"
        )
    return encoder.float().eval()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()


def main():
    args = parse_args()
    device = select_device(args.device)
    dtype = DTYPES[args.dtype]

    loader_cls = get_class_from_dynamic_module("processor.ImageLoader3D", args.model)
    volume = loader_cls().load_image(
        args.ct_path, normalize_mode="ct", anatomy_region=args.region
    )
    volume = torch.as_tensor(volume, dtype=torch.float32).reshape(1, 1, 192, 192, 192)

    encoder = load_encoder(args.model)
    with torch.inference_mode():
        ref = encoder(volume).double()

        encoder = encoder.to(device=device, dtype=dtype)
        x = volume.to(device=device, dtype=dtype)
        times = []
        for _ in range(args.runs + 1):
            start = time.perf_counter()
            out = encoder(x)
            sync(device)
            times.append(time.perf_counter() - start)
        out = out.double().cpu()

    ref = ref.reshape(-1, ref.shape[-1])
    out = out.reshape(-1, out.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(
        ref.flatten(), out.flatten(), dim=0
    ).item()
    token_cosine = torch.nn.functional.cosine_similarity(ref, out, dim=1)
    result = {
        "device": device,
        "device_name": torch.xpu.get_device_name() if device == "xpu" else device,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "tokens": list(out.shape),
        "one_minus_cosine": 1.0 - cosine,
        "one_minus_min_token_cosine": 1.0 - token_cosine.min().item(),
        "rel_l2": ((ref - out).norm() / ref.norm()).item(),
        "max_abs_diff": (ref - out).abs().max().item(),
        "first_run_s": times[0],
        "median_s": statistics.median(times[1:]) if args.runs else None,
    }
    print(json.dumps(result, indent=2))
    if cosine < args.min_cosine:
        print(
            f"cosine {cosine:.6f} is below --min-cosine {args.min_cosine}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
