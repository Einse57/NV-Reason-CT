#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Check OpenVINO encoder agreement against the PyTorch FP32 CPU reference.

For each input, the reference tokens come from upstream ``Vision3D`` in float32
on CPU. Each OpenVINO device then runs in its own subprocess so a device that
hangs while compiling can be stopped (``--compile-timeout``). CPU and GPU run
with ``INFERENCE_PRECISION_HINT=f32``; NPU uses its default precision.

Example:
    python openvino/scripts/agreement_encoder.py --model-dir /path/to/NV-Reason-CT \
        --ir openvino/ir/encoder_fp32/vision_fp32.xml --devices CPU GPU NPU \
        --input ct.nii.gz:chest --input synthetic:0 --out agreement.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from encoder_common import build_reference_encoder, compare, load_input


def ov_config(device: str, gpu_large_alloc: bool = False) -> dict:
    cfg = {"INFERENCE_PRECISION_HINT": "f32"} if device in ("CPU", "GPU") else {}
    if device == "GPU" and gpu_large_alloc:
        # f32 attention scores are 12 x 13824^2 x 4 B = 9.2 GB, above the
        # default 4 GB single-allocation limit on Intel GPUs.
        cfg["GPU_ENABLE_LARGE_ALLOCATIONS"] = "YES"
    return cfg


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill a worker and its children (a Windows venv python.exe is a launcher)."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        proc.kill()
    proc.communicate()


def worker(args) -> None:
    import openvino as ov

    core = ov.Core()
    t0 = time.perf_counter()
    compiled = core.compile_model(
        args.ir, args.device, ov_config(args.device, args.gpu_large_alloc)
    )
    compile_s = time.perf_counter() - t0
    x = np.load(args.in_npy)
    t1 = time.perf_counter()
    out = compiled.create_infer_request().infer({0: x})
    infer_s = time.perf_counter() - t1
    np.save(args.out_npy, np.asarray(next(iter(out.values())), dtype=np.float32))
    try:
        name = core.get_property(args.device, "FULL_DEVICE_NAME")
    except Exception:  # noqa: BLE001
        name = args.device
    prec = "default"
    if args.device != "NPU":
        hint = compiled.get_property("INFERENCE_PRECISION_HINT")
        prec = hint.get_type_name() if hasattr(hint, "get_type_name") else str(hint)
    print(
        json.dumps(
            {
                "compile_s": compile_s,
                "infer_s": infer_s,
                "device_name": name,
                "exec_precision": prec,
            }
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", help="local snapshot of nvidia/NV-Reason-CT")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--devices", nargs="+", default=["CPU"])
    ap.add_argument(
        "--input",
        action="append",
        default=[],
        help="synthetic[:seed] | file.npy | ct.nii.gz[:region] (repeatable)",
    )
    ap.add_argument("--work-dir", default="agreement_work")
    ap.add_argument(
        "--compile-timeout", type=float, default=900.0, help="seconds per device"
    )
    ap.add_argument("--out", default="agreement.json")
    ap.add_argument(
        "--gpu-large-alloc",
        action="store_true",
        help="set GPU_ENABLE_LARGE_ALLOCATIONS=YES (needed for f32 on Intel GPU)",
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--device", help=argparse.SUPPRESS)
    ap.add_argument("--in-npy", help=argparse.SUPPRESS)
    ap.add_argument("--out-npy", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return

    import openvino as ov

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    inputs = args.input or ["synthetic:0"]
    encoder = torch = None
    failed: dict[str, str] = {}
    report = {
        "openvino": ov.__version__,
        "host_cpu": platform.processor(),
        "ir": Path(args.ir).name,
        "results": [],
    }
    for k, spec in enumerate(inputs):
        path, _, region = (
            spec.partition(":")
            if spec.endswith((":chest", ":abdomen"))
            else (spec, "", "")
        )
        tag = f"in{k}"
        x = load_input(path, args.model_dir, region or "chest")
        in_npy = work / f"{tag}_input.npy"
        np.save(in_npy, x)
        ref_npy = work / f"{tag}_torch.npy"
        if ref_npy.exists():
            ref = np.load(ref_npy)
        else:
            if encoder is None:
                encoder, torch = build_reference_encoder(args.model_dir, args.weights)
            with torch.no_grad():
                t0 = time.perf_counter()
                ref = encoder(torch.from_numpy(x)).numpy()
                print(f"[{tag}] torch {time.perf_counter() - t0:.1f}s", flush=True)
            np.save(ref_npy, ref)
        label = Path(path).name if not path.startswith("synthetic") else path
        for dev in args.devices:
            out_npy = work / f"{tag}_{dev}.npy"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--ir",
                args.ir,
                "--device",
                dev,
                "--in-npy",
                str(in_npy),
                "--out-npy",
                str(out_npy),
            ]
            if args.gpu_large_alloc:
                cmd.append("--gpu-large-alloc")
            row = {
                "input": label,
                "region": region or None,
                "device": dev,
                "precision": "f32" if dev in ("CPU", "GPU") else "NPU default",
                "config": ov_config(dev, args.gpu_large_alloc),
            }
            if dev in failed:
                row.update(ok=False, error=f"skipped: {dev} failed on {failed[dev]}")
                report["results"].append(row)
                continue
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            try:
                stdout, stderr = proc.communicate(timeout=args.compile_timeout)
                if proc.returncode == 0:
                    row.update(json.loads(stdout.strip().splitlines()[-1]))
                    row.update(compare(ref, np.load(out_npy)))
                    row["ok"] = True
                else:
                    row["ok"] = False
                    row["error"] = (stderr or stdout).strip()[-2000:]
            except subprocess.TimeoutExpired:
                kill_tree(proc)
                row["ok"] = False
                row["error"] = (
                    f"timeout after {args.compile_timeout:.0f}s (compile + one inference)"
                )
            print(
                json.dumps({k2: v for k2, v in row.items() if k2 != "error"}),
                flush=True,
            )
            if not row["ok"]:
                failed[dev] = label
                print("  error:", row["error"][-400:], flush=True)
            report["results"].append(row)
            Path(args.out).write_text(json.dumps(report, indent=1))
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
