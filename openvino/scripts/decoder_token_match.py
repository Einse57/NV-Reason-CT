#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1
"""Experimental greedy token-match check: OpenVINO text tower vs PyTorch.

Both backends get the same ``inputs_embeds``: text-token embeddings from the
checkpoint with the 13,824 image-pad positions replaced by encoder tokens
(``[13824, 2560]`` from the PyTorch FP32 encoder, saved as ``.npy``). The
optimum-intel text export only takes ``input_ids``, so a Parameter
``inputs_embeds`` is spliced in place of the embedding Gather. Positions are
1-D text positions on both sides because the text-only export has no 3D
(M-RoPE) position input, so this measures backend agreement and is not a full
NV-Reason-CT report reproduction.

Run the two stages in separate processes, then compare:
    python decoder_token_match.py --stage prep  --model-dir NV-Reason-CT --ct ct.nii.gz --encoder-npy tok.npy --work w
    python decoder_token_match.py --stage torch --lm-dir hf_lm_only --work w
    python decoder_token_match.py --stage ov    --ir decoder_int8/openvino_model.xml --work w
    python decoder_token_match.py --stage compare --lm-dir hf_lm_only --work w --out token_match.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROMPTS = {
    "chest": "write a structured chest CT report",
    "abdomen": "write a structured abdominal CT report",
}


def stage_prep(args, work: Path) -> None:
    import torch
    from safetensors import safe_open
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPTS[args.region]},
            ],
        }
    ]
    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = proc(
        text=text, images3d=[args.ct], anatomy_region=args.region, return_tensors="pt"
    )
    ids = enc["input_ids"][0]
    image_id = json.loads((Path(args.model_dir) / "config.json").read_text())[
        "image_token_id"
    ]
    mask = (ids == image_id).numpy()
    vis = np.load(args.encoder_npy).astype(np.float32)
    assert mask.sum() == vis.shape[0], (mask.sum(), vis.shape)
    with safe_open(
        str(Path(args.model_dir) / "model.safetensors"), framework="pt"
    ) as f:
        emb = f.get_tensor("model.language_model.embed_tokens.weight").float().numpy()
    embeds = emb[ids.numpy()]
    embeds[mask] = vis
    np.save(work / "embed_table.npy", emb)
    np.save(work / "input_ids.npy", ids.numpy())
    np.save(work / "inputs_embeds.npy", embeds[None].astype(np.float32))
    print(
        "prompt tokens",
        len(ids),
        "image tokens",
        int(mask.sum()),
        "dtype",
        torch.float32,
    )


def stage_torch(args, work: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    embeds = torch.from_numpy(np.load(work / "inputs_embeds.npy"))
    model = AutoModelForCausalLM.from_pretrained(
        args.lm_dir, dtype=torch.float32
    ).eval()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            inputs_embeds=embeds,
            attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long),
            max_new_tokens=args.new_tokens,
            do_sample=False,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    toks = (
        out.sequences[0, -args.new_tokens :].numpy()
        if out.sequences.shape[1] >= args.new_tokens
        else out.sequences[0].numpy()
    )
    np.save(work / "torch_tokens.npy", toks)
    np.save(work / "torch_first_logits.npy", out.scores[0][0].float().numpy())
    print("torch", time.perf_counter() - t0, "s", toks.tolist())


def splice_inputs_embeds(model):
    import openvino as ov
    from openvino import opset13 as ops

    ids = next(
        p
        for p in model.get_parameters()
        if p.get_output_tensor(0).get_names() & {"input_ids"}
    )

    def from_ids(node, depth=0):
        if node.get_friendly_name() == ids.get_friendly_name():
            return True
        if depth > 4 or node.get_type_name() not in ("Convert", "Parameter"):
            return False
        return any(
            from_ids(i.get_source_output().get_node(), depth + 1) for i in node.inputs()
        )

    gathers = [
        n
        for n in model.get_ordered_ops()
        if n.get_type_name() == "Gather"
        and from_ids(n.input(1).get_source_output().get_node())
        and n.get_output_partial_shape(0).rank.get_length() == 3
    ]
    assert len(gathers) == 1, [g.get_friendly_name() for g in gathers]
    g = gathers[0]
    hidden = g.get_output_partial_shape(0)[2].get_length()
    p = ops.parameter(
        ov.PartialShape([-1, -1, hidden]),
        g.get_output_element_type(0),
        name="inputs_embeds",
    )
    p.get_output_tensor(0).set_names({"inputs_embeds"})
    for target in list(g.output(0).get_target_inputs()):
        target.replace_source_output(p.output(0))
    model.add_parameters([p])
    model.validate_nodes_and_infer_types()
    return model


def stage_ov(args, work: Path) -> None:
    import openvino as ov

    core = ov.Core()
    model = splice_inputs_embeds(core.read_model(args.ir))
    cfg = {"INFERENCE_PRECISION_HINT": "f32"} if args.device in ("CPU", "GPU") else {}
    t0 = time.perf_counter()
    compiled = core.compile_model(model, args.device, cfg)
    compile_s = time.perf_counter() - t0
    req = compiled.create_infer_request()
    emb = np.load(work / "embed_table.npy", mmap_mode="r")
    embeds = np.load(work / "inputs_embeds.npy")
    L = embeds.shape[1]
    req.reset_state()
    toks, first_logits = [], None
    feed = {
        "inputs_embeds": embeds,
        "input_ids": np.zeros((1, L), np.int64),
        "attention_mask": np.ones((1, L), np.int64),
        "position_ids": np.arange(L, dtype=np.int64)[None],
        "beam_idx": np.zeros(1, np.int32),
    }
    t1 = time.perf_counter()
    for step in range(args.new_tokens):
        logits = req.infer(feed)["logits"][0, -1]
        if first_logits is None:
            first_logits = np.array(logits, dtype=np.float32)
            prefill_s = time.perf_counter() - t1
        tok = int(np.argmax(logits))
        toks.append(tok)
        n = L + step
        feed = {
            "inputs_embeds": np.asarray(emb[tok], np.float32)[None, None],
            "input_ids": np.zeros((1, 1), np.int64),
            "attention_mask": np.ones((1, n + 1), np.int64),
            "position_ids": np.array([[n]], np.int64),
            "beam_idx": np.zeros(1, np.int32),
        }
    np.save(work / "ov_tokens.npy", np.array(toks))
    np.save(work / "ov_first_logits.npy", first_logits)
    info = {
        "device": args.device,
        "compile_s": compile_s,
        "prefill_s": prefill_s,
        "total_generate_s": time.perf_counter() - t1,
        "openvino": ov.__version__,
        "precision_hint": cfg.get("INFERENCE_PRECISION_HINT", "default"),
    }
    (work / "ov_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), toks)


def stage_compare(args, work: Path) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.lm_dir)
    t, o = np.load(work / "torch_tokens.npy"), np.load(work / "ov_tokens.npy")
    n = min(len(t), len(o))
    eq = t[:n] == o[:n]
    first_div = int(np.argmin(eq)) if not eq.all() else None
    lt, lo = (
        np.load(work / "torch_first_logits.npy").astype(np.float64),
        np.load(work / "ov_first_logits.npy").astype(np.float64),
    )
    res = {
        "new_tokens": int(n),
        "position_match": float(eq.mean()),
        "first_divergence": first_div,
        "prefix_match_len": int(n if first_div is None else first_div),
        "first_step_logits_cosine": float(
            lt @ lo / (np.linalg.norm(lt) * np.linalg.norm(lo))
        ),
        "first_step_top1_equal": bool(lt.argmax() == lo.argmax()),
        "first_step_top5_overlap": len(
            set(np.argsort(-lt)[:5]) & set(np.argsort(-lo)[:5])
        ),
        "torch_text": tok.decode(t[:n]),
        "ov_text": tok.decode(o[:n]),
    }
    info = work / "ov_info.json"
    if info.exists():
        res["ov"] = json.loads(info.read_text())
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage", choices=["prep", "torch", "ov", "compare"], required=True
    )
    ap.add_argument("--work", default="token_match_work")
    ap.add_argument("--model-dir")
    ap.add_argument("--lm-dir")
    ap.add_argument("--ct")
    ap.add_argument("--region", default="chest", choices=list(PROMPTS))
    ap.add_argument("--encoder-npy")
    ap.add_argument("--ir")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--out", default="token_match.json")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    {
        "prep": stage_prep,
        "torch": stage_torch,
        "ov": stage_ov,
        "compare": stage_compare,
    }[args.stage](args, work)


if __name__ == "__main__":
    main()
