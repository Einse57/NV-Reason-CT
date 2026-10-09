#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Time one NV-Reason-CT encoder backend in this process.

Backends: ``torch`` (upstream Vision3D on ``--torch-device`` cpu or xpu, in
``--torch-dtype``, optionally ``torch.compile``) or ``ov`` (OpenVINO IR on
``--device``; CPU/GPU pinned to f32 unless ``--ov-precision`` is given). Reports
load/compile time, first inference, mean/p50/min over ``--runs`` and the peak
RSS of the process. Run one backend per process so peak RSS is per backend.

Example:
    python openvino/scripts/bench_encoder.py --backend ov --device CPU \
        --ir openvino/ir/encoder_fp32/vision_fp32.xml --runs 5 --out ov_cpu.json
    python openvino/scripts/bench_encoder.py --backend torch --torch-device xpu \
        --torch-dtype fp16 --model-dir /path/to/NV-Reason-CT --runs 5
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from encoder_common import (
    INPUT_SHAPE,
    compare,
    build_reference_encoder,
    load_input,
    peak_rss_mb,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["torch", "ov"], required=True)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--ir", help="OpenVINO IR (.xml) for --backend ov")
    ap.add_argument("--model-dir", help="model snapshot for --backend torch")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--input", default="synthetic:0", help="synthetic[:seed] or .npy")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--torch-device", default="cpu", help="cpu or xpu (--backend torch)")
    ap.add_argument(
        "--torch-dtype",
        choices=["f32", "fp16", "bf16"],
        default="f32",
        help="cast the whole encoder and input (--backend torch)",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the encoder; runs under no_grad (see RESULTS.md)",
    )
    ap.add_argument("--ov-precision", default="f32", help="INFERENCE_PRECISION_HINT")
    ap.add_argument("--ov-threads", type=int, default=0, help="INFERENCE_NUM_THREADS")
    ap.add_argument("--ref", default=None, help="reference tokens .npy for agreement")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--gpu-large-alloc",
        action="store_true",
        help="GPU_ENABLE_LARGE_ALLOCATIONS=YES",
    )
    args = ap.parse_args()

    x = load_input(args.input)
    res = {
        "backend": args.backend,
        "device": args.device,
        "input_shape": list(INPUT_SHAPE),
        "host_cpu": platform.processor(),
        "runs": args.runs,
    }
    t0 = time.perf_counter()
    if args.backend == "torch":
        encoder, torch = build_reference_encoder(args.model_dir, args.weights)
        dtype = {"f32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            args.torch_dtype
        ]
        dev = torch.device(args.torch_device)
        encoder = encoder.to(device=dev, dtype=dtype)
        xt = torch.from_numpy(x).to(device=dev, dtype=dtype)
        res.update(
            torch=torch.__version__,
            torch_threads=torch.get_num_threads(),
            precision=args.torch_dtype,
            compile=args.compile,
        )
        if dev.type == "xpu":
            res["device"] = "xpu"
            res["device_name"] = torch.xpu.get_device_name(dev)
        if args.compile:
            encoder = torch.compile(encoder)
        # torch.compile + inference_mode hits a Dynamo guard error on this
        # encoder (drop-path attribute); no_grad works for both paths.
        grad_ctx = torch.no_grad

        def sync():
            if dev.type == "xpu":
                torch.xpu.synchronize()

        def run():
            with grad_ctx():
                y = encoder(xt)
            sync()
            return y.float().cpu().numpy()
    else:
        import openvino as ov

        core = ov.Core()
        cfg = (
            {"INFERENCE_PRECISION_HINT": args.ov_precision}
            if args.device in ("CPU", "GPU")
            else {}
        )
        if args.device == "CPU" and args.ov_threads:
            cfg["INFERENCE_NUM_THREADS"] = args.ov_threads
        if args.device == "GPU" and args.gpu_large_alloc:
            cfg["GPU_ENABLE_LARGE_ALLOCATIONS"] = "YES"
        compiled = core.compile_model(args.ir, args.device, cfg)
        req = compiled.create_infer_request()
        res.update(
            openvino=ov.__version__,
            precision=cfg.get("INFERENCE_PRECISION_HINT", "default"),
            config=cfg,
            device_name=core.get_property(args.device, "FULL_DEVICE_NAME"),
        )
        if args.device == "CPU":
            res["ov_threads_used"] = int(compiled.get_property("INFERENCE_NUM_THREADS"))

        def run():
            return next(iter(req.infer({0: x}).values()))

    res["load_or_compile_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    out = run()
    res["first_run_s"] = time.perf_counter() - t1
    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        out = run()
        times.append(time.perf_counter() - t)
    if args.ref:
        res["agreement"] = compare(np.load(args.ref), np.asarray(out))
    res.update(
        times_s=times,
        mean_s=statistics.mean(times),
        p50_s=statistics.median(times),
        min_s=min(times),
        peak_rss_mb=peak_rss_mb(),
    )
    text = json.dumps(res, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
