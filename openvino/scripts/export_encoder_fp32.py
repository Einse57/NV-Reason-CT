#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Export the NV-Reason-CT 3D encoder (Primus ViT + merger) to FP32 OpenVINO IR.

Input ``volume`` [1, 1, 192, 192, 192] float32 -> output ``tokens`` [13824, 2560].
Weights are kept in FP32 (``compress_to_fp16=False``). Conversion goes
straight from the PyTorch module (no ONNX step); peak RSS measured about 3 GB
with OpenVINO 2026.4.1 and torch 2.14.1.

Example:
    python openvino/scripts/export_encoder_fp32.py --model-dir /path/to/NV-Reason-CT \
        --out-dir openvino/ir/encoder_fp32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from encoder_common import INPUT_SHAPE, build_reference_encoder, peak_rss_mb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model-dir", required=True, help="local snapshot of nvidia/NV-Reason-CT"
    )
    ap.add_argument(
        "--weights",
        default=None,
        help="optional safetensors with vision3d tensors only",
    )
    ap.add_argument("--out-dir", default="openvino/ir/encoder_fp32")
    args = ap.parse_args()

    import openvino as ov

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    encoder, torch = build_reference_encoder(args.model_dir, args.weights)
    example = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    t1 = time.perf_counter()
    with torch.no_grad():
        ov_model = ov.convert_model(
            encoder, example_input=example, input=[(list(INPUT_SHAPE), ov.Type.f32)]
        )
    ov_model.inputs[0].get_tensor().set_names({"volume"})
    ov_model.outputs[0].get_tensor().set_names({"tokens"})
    t2 = time.perf_counter()
    xml = out / "vision_fp32.xml"
    ov.save_model(ov_model, str(xml), compress_to_fp16=False)
    t3 = time.perf_counter()
    info = {
        "openvino": ov.__version__,
        "torch": torch.__version__,
        "input": list(INPUT_SHAPE),
        "output": str(ov_model.outputs[0].get_partial_shape()),
        "load_s": round(t1 - t0, 2),
        "convert_s": round(t2 - t1, 2),
        "save_s": round(t3 - t2, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "bin_mb": round(xml.with_suffix(".bin").stat().st_size / 2**20, 1),
    }
    (out / "export_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
