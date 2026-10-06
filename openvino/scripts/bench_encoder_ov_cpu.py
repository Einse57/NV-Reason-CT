#!/usr/bin/env python3
"""Box-only / non-target OpenVINO CPU encoder latency + peak RSS bench.

Published NVIDIA baseline (RTX PRO 6000): ~25 s/report for full NV-Reason-CT.
This script benches ONLY the 3D vision encoder on CPU and must not be compared
directly to that end-to-end GPU number.
"""
from __future__ import annotations
import argparse, json, os, resource, time
from pathlib import Path
import numpy as np
import openvino as ov

def peak_rss_mb() -> float:
    # Linux ru_maxrss is kilobytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ir/encoder_from_onnx/vision.xml")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="benches/encoder_ov_cpu.json")
    args = ap.parse_args()

    print("LABEL: box-only / non-target (OpenVINO CPU encoder only)")
    print("NOTE: NVIDIA published ~25 s/report on RTX PRO 6000 for full VLM; this is encoder-only CPU.")
    core = ov.Core()
    model = core.read_model(args.model)
    compiled = core.compile_model(model, "CPU")
    req = compiled.create_infer_request()
    volume = np.random.randn(1, 1, 192, 192, 192).astype(np.float32) * 0.1

    for _ in range(args.warmup):
        req.infer({0: volume})

    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        req.infer({0: volume})
        dt = time.perf_counter() - t0
        times.append(dt)
        out = req.get_output_tensor(0).data
        print(f"run {i}: {dt*1000:.1f} ms  out={out.shape}")

    rss = peak_rss_mb()
    summary = {
        "label": "box-only / non-target",
        "device": "CPU",
        "component": "encoder_only",
        "model": args.model,
        "input_shape": [1, 1, 192, 192, 192],
        "output_shape": list(out.shape),
        "warmup": args.warmup,
        "runs": args.runs,
        "latency_s": times,
        "latency_mean_s": float(np.mean(times)),
        "latency_std_s": float(np.std(times)),
        "latency_mean_ms": float(np.mean(times) * 1000),
        "peak_rss_mb": rss,
        "published_baseline": "RTX PRO 6000 ~25 s/report (full VLM, not comparable)",
        "host": os.uname().nodename,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
