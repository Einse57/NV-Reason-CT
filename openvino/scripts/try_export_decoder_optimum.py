#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Experimental: export the NV-Reason-CT Qwen3.5 text tower with optimum-intel.

Run ``extract_text_tower.py`` first; it writes a ``qwen3_5_text`` checkpoint
(``Qwen3_5ForCausalLM``). optimum-intel registers ``qwen3_5`` only for
``image-text-to-text`` (huggingface/optimum-intel#1721), so the text-only
checkpoint is exported with ``text-generation-with-past``. INT8 weight
compression is the default; INT4 on hybrid Qwen3.5 is known to give incoherent
output (huggingface/optimum-intel#1722).

Example:
    python openvino/scripts/try_export_decoder_optimum.py -m hf_lm_only \
        --out openvino/ir/decoder_int8 --weight-format int8
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def versions() -> dict:
    out = {}
    for pkg in (
        "optimum-intel",
        "optimum",
        "transformers",
        "openvino",
        "nncf",
        "torch",
    ):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "-m", "--model", required=True, help="text-only qwen3_5_text checkpoint dir"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="text-generation-with-past")
    ap.add_argument(
        "--weight-format", default="int8", choices=["fp32", "fp16", "int8", "int4"]
    )
    args = ap.parse_args()

    from optimum.exporters.openvino import main_export

    info = {
        "versions": versions(),
        "task": args.task,
        "weight_format": args.weight_format,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        main_export(
            model_name_or_path=args.model,
            output=args.out,
            task=args.task,
            weight_format=args.weight_format,
            library_name="transformers",
        )
        info["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - record the exact failure
        info["status"] = "failed"
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["traceback"] = traceback.format_exc()[-4000:]
    info["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (Path(args.out) / "export_status.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "traceback"}, indent=1))
    if info["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
