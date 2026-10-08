#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Extract the NV-Reason-CT Qwen3.5 language tower as a text-only checkpoint.

Writes ``model.safetensors`` (``model.language_model.*`` renamed to ``model.*``,
plus ``lm_head.weight``), a ``qwen3_5_text`` config with
``architectures=["Qwen3_5ForCausalLM"]`` and the tokenizer files. The 3D and 2D
vision towers are dropped. The result can be passed to optimum-intel with
``--task text-generation-with-past`` (see huggingface/optimum-intel#1721).
Needs about 11 GB of free RAM and disk.

Example:
    python openvino/scripts/extract_text_tower.py --model-dir /path/to/NV-Reason-CT --out-dir hf_lm_only
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file

    src, out = Path(args.model_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    text_cfg = dict(cfg["text_config"])
    text_cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    text_cfg["model_type"] = "qwen3_5_text"
    text_cfg["transformers_version"] = cfg.get("transformers_version")
    (out / "config.json").write_text(json.dumps(text_cfg, indent=2))

    state = {}
    with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not a dict
            if key.startswith("model.language_model."):
                state["model." + key[len("model.language_model.") :]] = f.get_tensor(
                    key
                )
            elif key == "lm_head.weight":
                state[key] = f.get_tensor(key)
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    for name in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "LICENSE",
    ):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    print(f"wrote {len(state)} tensors to {out}")


if __name__ == "__main__":
    main()
