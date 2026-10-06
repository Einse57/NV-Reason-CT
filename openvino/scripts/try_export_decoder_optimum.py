#!/usr/bin/env python3
"""Attempt optimum-intel export of NV-Reason-CT language tower (Qwen3.5 hybrid).

Prereqs (typically ≥32–64 GiB RAM):
  1. Extract language_model (+lm_head) from nvidia/NV-Reason-CT model.safetensors
     into a folder with model_type=qwen3_5_text (see docs).
  2. transformers≈5.6.x, recent optimum-intel (git main recommended), openvino, nncf.

On 15 GiB boxes this will likely OOM. Community ONNX decoder cannot be read by
OpenVINO (see openvino/BLOCKER.md).
"""
from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="HF id or local qwen3_5 / qwen3_5_text dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="image-text-to-text",
                    help="optimum-intel currently registers qwen3_5 under image-text-to-text; "
                         "text-only may need qwen3_5_text + text-generation-with-past on newer builds")
    ap.add_argument("--weight-format", default="int4")
    args = ap.parse_args()
    from optimum.exporters.openvino import main_export
    Path(args.out).mkdir(parents=True, exist_ok=True)
    main_export(
        model_name_or_path=args.model,
        output=args.out,
        task=args.task,
        weight_format=args.weight_format,
        trust_remote_code=True,
        library_name="transformers",
    )
    print("DECODER_EXPORT_OK", args.out)

if __name__ == "__main__":
    main()
