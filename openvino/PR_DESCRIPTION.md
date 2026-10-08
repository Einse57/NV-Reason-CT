# OpenVINO tooling for the NV-Reason-CT encoder, plus an experimental text-tower export

Draft on the Einse57 fork only.

## Summary

- `scripts/export_encoder_fp32.py`: exports the upstream 3D encoder (Primus ViT + merger) from `nvidia/NV-Reason-CT` to FP32 OpenVINO IR (`volume` [1,1,192,192,192] → `tokens` [13824,2560]), directly from PyTorch.
- `scripts/agreement_encoder.py`: compares OpenVINO devices with the PyTorch FP32 CPU reference on real CT and synthetic input. Each device runs in its own process with a compile timeout.
- `scripts/bench_encoder.py`: times one backend per process (load/compile, first run, mean/p50/min, peak RSS).
- `scripts/extract_text_tower.py`, `scripts/try_export_decoder_optimum.py`, `scripts/decoder_token_match.py`: experimental INT8 export of the Qwen3.5 text tower with optimum-intel, plus a greedy token-match check against PyTorch.
- `tests/test_encoder_openvino_agreement.py`: a small random-init Primus round trip through OpenVINO. It is skipped when openvino or torch is not installed.
- Optional dependency files: `requirements-openvino.txt` and `requirements-decoder-export.txt`. The default PyTorch inference path is unchanged.
- Removed the earlier community-ONNX encoder scripts and box bench files, which the FP32 path replaces.

Research and development use only; not for clinical diagnosis. The IR is derived from OpenMDW-1.1 model materials, so it carries the same license and notices (`docs/LICENSE_AND_DISTRIBUTION.md`). No weights or IR are committed.

## Hardware tested

- Intel Core Ultra 9 285H, 64 GB: CPU, Arc 140T iGPU, NPU
- Intel Core Ultra 7 265F, 64 GB: CPU, NPU

Software: OpenVINO 2026.4.1, torch 2.14.1+cpu, transformers 5.6.2 (for the decoder export, 5.2.0 with optimum-intel 2.2.0), Windows 11.

Test CT: TotalSegmentator CT v2.0.1, case s0050 (CC BY 4.0, doi:10.5281/zenodo.10047292), with chest and abdomen crops from the upstream `ImageLoader3D`.

## Encoder agreement vs PyTorch FP32 CPU

| Host | Device | Input | Precision | 1 − cosine | 1 − min per-token cosine | Max abs diff | Rel. L2 | Compile (s) |
|---|---|---|---|---|---|---|---|---|
| Core Ultra 9 285H (64 GB) | CPU | CT s0050, chest crop | f32 | 6.7e-13 | 5.6e-12 | 2.9e-06 | 1.1e-06 | 2.4 |
| Core Ultra 9 285H (64 GB) | CPU | CT s0050, abdomen crop | f32 | 6.7e-13 | 3.3e-12 | 3.1e-06 | 1.1e-06 | 2.4 |
| Core Ultra 9 285H (64 GB) | CPU | synthetic U(-1,1), seed 0 | f32 | 4.4e-13 | 6.5e-13 | 1.8e-06 | 8.3e-07 | 2.4 |
| Core Ultra 9 285H (64 GB) | Arc 140T iGPU | CT s0050, chest crop | f32 | 1.1e-11 | 4.2e-10 | 2.7e-05 | 4.6e-06 | 8.6 |
| Core Ultra 9 285H (64 GB) | Arc 140T iGPU | CT s0050, abdomen crop | f32 | 8.7e-12 | 2.6e-10 | 1.8e-05 | 4.2e-06 | 6.4 |
| Core Ultra 9 285H (64 GB) | Arc 140T iGPU | synthetic U(-1,1), seed 0 | f32 | 2.3e-12 | 1.9e-11 | 4.7e-06 | 2.1e-06 | 6.4 |
| Core Ultra 9 285H (64 GB) | NPU | CT s0050, chest crop | NPU default | — | — | — | — | not compiled: timeout after 900s (compile + one inference) |
| Core Ultra 7 265F (64 GB) | CPU | CT s0050, chest crop | f32 | 6.5e-13 | 5.5e-12 | 3.6e-06 | 1.1e-06 | 2.4 |
| Core Ultra 7 265F (64 GB) | CPU | CT s0050, abdomen crop | f32 | 6.6e-13 | 3.4e-12 | 3.1e-06 | 1.1e-06 | 2.3 |
| Core Ultra 7 265F (64 GB) | CPU | synthetic U(-1,1), seed 0 | f32 | 4.2e-13 | 6.4e-13 | 1.8e-06 | 8.3e-07 | 2.3 |
| Core Ultra 7 265F (64 GB) | NPU | CT s0050, chest crop | NPU default | — | — | — | — | not compiled: timeout after 900s (compile + one inference) |

CPU and GPU used `INFERENCE_PRECISION_HINT=f32`. The GPU also needs `GPU_ENABLE_LARGE_ALLOCATIONS=YES`: the f32 attention scores need a single 9.2 GB allocation, and the default limit is 4 GB. The NPU compile did not finish within 15 minutes on either host, and it produced no error message.

## Encoder timing (interactive desktop session, CT chest crop, 5 timed runs)

| Host | Backend / device | Precision | Load or compile (s) | First run (s) | Mean (s) | p50 (s) | Min (s) | Peak RSS (GiB) |
|---|---|---|---|---|---|---|---|---|
| Core Ultra 9 285H (64 GB) | PyTorch, CPU (16 threads) | f32 | 10.2 | 24.4 | 23.8 | 23.9 | 23.7 | 1.7 |
| Core Ultra 9 285H (64 GB) | OpenVINO, CPU | f32 | 2.6 | 28.8 | 28.6 | 28.6 | 28.3 | 7.7 |
| Core Ultra 9 285H (64 GB) | OpenVINO, Arc 140T iGPU | f32 | 5.6 | 17.4 | 17.4 | 17.4 | 17.3 | 20.4 |
| Core Ultra 9 285H (64 GB) | OpenVINO, NPU | default | not compiled within 15 min | — | — | — | — | — |
| Core Ultra 7 265F (64 GB) | PyTorch, CPU (20 threads) | f32 | 5.3 | 12.7 | 16.4 | 16.4 | 16.1 | 1.7 |
| Core Ultra 7 265F (64 GB) | OpenVINO, CPU | f32 | 3.0 | 21.6 | 21.9 | 21.9 | 21.8 | 7.7 |
| Core Ultra 7 265F (64 GB) | OpenVINO, NPU | default | not compiled within 15 min | — | — | — | — | — |

## Text tower (experimental)

- The optimum-intel 2.2.0 `text-generation-with-past` export of the extracted `qwen3_5_text` tower with INT8 weights works with transformers 5.2.0 (4.2 GB IR). With transformers 5.5.4 it fails with "Maximum required is 5.2.*" (huggingface/optimum-intel#1721 covers the missing `qwen3_5` text route).
- Token match on the s0050 chest case, using the same `inputs_embeds` (13,844 tokens, including the 13,824 encoder tokens): 32/32 greedy tokens identical to PyTorch FP32 on the 265F CPU. First-step logits cosine was 0.9985.
- Limit: the text-only export takes 1-D positions, while the full model uses 3-D M-RoPE positions for image tokens, so this result is backend agreement, not end-to-end report generation. Details are in `docs/DECODER.md`. INT4 was not tried (huggingface/optimum-intel#1722).

Full write-up: `docs/RESULTS.md`, with machine-readable results in `results/`.
