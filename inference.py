# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run NV-Reason-CT inference on one NIfTI CT volume."""

import argparse
import sys

import torch
from transformers import (
    AttentionInterface,
    AttentionMaskInterface,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask


DEFAULT_MODEL = "nvidia/NV-Reason-CT"
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def select_device(requested):
    """Return the requested device, or the first available of cuda, xpu, cpu."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def sdpa_xpu_attention_forward(module, query, key, value, attention_mask, **kwargs):
    """SDPA that runs decoding steps in float32 on XPU.

    With torch 2.14.1+xpu, half-precision SDPA returns wrong values when the
    head dimension is above 128 and the query is shorter than the key, which is
    every decoding step of the 256-dim full-attention layers. Prefill is not
    affected. Decoding steps are small, so they run in float32 here.
    """
    if query.dtype == torch.float32 or query.shape[-2] == key.shape[-2]:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.float()
    output, weights = sdpa_attention_forward(
        module, query.float(), key.float(), value.float(), attention_mask, **kwargs
    )
    return output.to(query.dtype), weights


AttentionInterface.register("sdpa_xpu", sdpa_xpu_attention_forward)
AttentionMaskInterface.register("sdpa_xpu", sdpa_mask)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ct_path", help="Path to a .nii or .nii.gz CT volume")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Local model directory or Hugging Face model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--region",
        choices=("chest", "abdomen"),
        default="chest",
        help="Anatomy-aware crop to use (default: chest)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Question or instruction (default: a structured report for the selected region)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking mode",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "xpu", "cpu"),
        default="auto",
        help="Device to run on (default: auto, the first available of cuda, xpu, cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", *DTYPES),
        default="auto",
        help="Model dtype (default: auto, bfloat16 on cuda/xpu and float32 on cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    if args.dtype == "auto":
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    else:
        dtype = DTYPES[args.dtype]
    if args.device == "auto" and device == "cpu":
        print(
            "No CUDA or XPU device found; running on CPU, which is slow.",
            file=sys.stderr,
        )

    default_prompts = {
        "chest": "write a structured chest CT report",
        "abdomen": "write a structured abdominal CT report",
    }
    prompt_text = args.prompt or default_prompts[args.region]

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa_xpu" if device == "xpu" else "sdpa",
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not args.disable_thinking,
    )
    inputs = processor(
        text=text,
        images3d=[args.ct_path],
        anatomy_region=args.region,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(response)


if __name__ == "__main__":
    main()
