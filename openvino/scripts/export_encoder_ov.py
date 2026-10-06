#!/usr/bin/env python3
"""Convert NV-Reason-CT 3D vision encoder to OpenVINO IR.

Preferred validated path on a 15 GiB box (2026-10-06):
  Community int8 ONNX (jarrelscy/nv-reason-ct-onnx vision.onnx + vision.data0)
  → ov.convert_model → IR

Upstream Primus FP32 torch→ONNX/OV OOMs on 15 GiB during tracing; use
`export_encoder_minimal.py` on a ≥32 GiB machine for FP32 IR from
nvidia/NV-Reason-CT `model.safetensors` (keys `model.vision3d.*`).

Agreement (zero volume): upstream FP32 torch vs community-int8 OV cosine ≈ 0.9997.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import openvino as ov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to vision.onnx (external data alongside)")
    ap.add_argument("--out-dir", default="openvino/ir/encoder")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(args.onnx)
    ov.save_model(model, str(out / "vision.xml"))
    print("Wrote", out / "vision.xml")
    core = ov.Core()
    compiled = core.compile_model(str(out / "vision.xml"), "CPU")
    print("CPU compile OK; inputs:", [(i.get_any_name(), i.shape) for i in compiled.inputs])
    print("outputs:", [(o.get_any_name(), o.shape) for o in compiled.outputs])

if __name__ == "__main__":
    main()
