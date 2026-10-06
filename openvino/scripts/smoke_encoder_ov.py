#!/usr/bin/env python3
"""CPU smoke: dummy 192³ volume through OpenVINO encoder; optional torch agreement."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import openvino as ov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvino/ir/encoder/vision.xml")
    ap.add_argument("--torch-ref", default=None, help="Optional .npy of torch tokens [13824,2560]")
    args = ap.parse_args()
    core = ov.Core()
    compiled = core.compile_model(args.model, "CPU")
    req = compiled.create_infer_request()
    vol = np.zeros((1, 1, 192, 192, 192), np.float32)
    req.infer({0: vol})
    out = req.get_output_tensor(0).data
    print("out", out.shape, "mean", float(out.mean()), "std", float(out.std()))
    if args.torch_ref:
        ref = np.load(args.torch_ref)
        a, b = ref.astype(np.float64).ravel(), out.astype(np.float64).ravel()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"cosine vs torch_ref = {cos:.8f}")
    print("SMOKE_OK")

if __name__ == "__main__":
    main()
