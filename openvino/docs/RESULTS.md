# Encoder and decoder results

All numbers below were measured on 2026-10-07 with OpenVINO 2026.4.1,
torch 2.14.1+cpu and transformers 5.6.2 (decoder export: 5.2.0) on Windows 11.
Machine-readable copies are in [`../results/`](../results/).

Hardware tested:

- Intel Core Ultra 9 285H, 64 GB RAM: CPU, Intel Arc 140T iGPU, NPU
- Intel Core Ultra 7 265F, 64 GB RAM: CPU, NPU

## Test input

- CT: TotalSegmentator CT dataset v2.0.1, case `s0050` (thorax-abdomen CT,
  Siemens Emotion 16, 130 kVp; 261 × 182 × 399 at 1.5 mm). Zenodo
  [10.5281/zenodo.10047292](https://doi.org/10.5281/zenodo.10047292), license
  CC BY 4.0. Wasserthal J, et al. "TotalSegmentator: Robust Segmentation of 104
  Anatomic Structures in CT Images." *Radiology: Artificial Intelligence*
  2023;5(5):e230024. The volume is not redistributed here.
- Preprocessing: upstream `ImageLoader3D` (CT normalization, LPS, 2 mm,
  192³), `chest` and `abdomen` anatomy crops. Both hosts produced
  bit-identical inputs.
- Synthetic: uniform in [-1, 1], seed 0.

## Encoder export

`export_encoder_fp32.py` produced a bit-identical FP32 IR on both hosts
(`vision_fp32.bin`, 596,158,924 bytes; 147.0 M parameters; static input
[1, 1, 192, 192, 192], output [13824, 2560]). Peak RSS during conversion was
3.0–3.2 GiB.

## Encoder agreement vs PyTorch FP32 (CPU)

Reference: upstream `Vision3D` in float32 on the same host's CPU.
CPU and GPU ran with `INFERENCE_PRECISION_HINT=f32`; the GPU also needs
`GPU_ENABLE_LARGE_ALLOCATIONS=YES` (without it, compilation fails with
`[GPU] Exceeded max size of memory object allocation: requested 9172942848
bytes, but max alloc size supported by device is 4294959104 bytes`).

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

**NPU.** On both hosts the NPU compile did not finish within the 15-minute cap,
either over SSH or in the interactive desktop session. When stopped, the
compiling process held about 16 GB (285H) and 19 GB (265F) of memory. No NPU error
message was produced. The encoder runs full attention over 13,824 tokens
(16 `ScaledDotProductAttention` ops in the IR).

## Encoder timing

One process per backend, run as a one-shot scheduled task in the interactive
desktop session. Input: CT s0050 chest crop. Load or compile, one first run,
then 5 timed runs. Peak RSS is the process peak working set (for the iGPU it
includes shared GPU memory).

| Host | Backend / device | Precision | Load or compile (s) | First run (s) | Mean (s) | p50 (s) | Min (s) | Peak RSS (GiB) |
|---|---|---|---|---|---|---|---|---|
| Core Ultra 9 285H (64 GB) | PyTorch, CPU (16 threads) | f32 | 10.2 | 24.4 | 23.8 | 23.9 | 23.7 | 1.7 |
| Core Ultra 9 285H (64 GB) | OpenVINO, CPU | f32 | 2.6 | 28.8 | 28.6 | 28.6 | 28.3 | 7.7 |
| Core Ultra 9 285H (64 GB) | OpenVINO, Arc 140T iGPU | f32 | 5.6 | 17.4 | 17.4 | 17.4 | 17.3 | 20.4 |
| Core Ultra 9 285H (64 GB) | OpenVINO, NPU | default | not compiled within 15 min | — | — | — | — | — |
| Core Ultra 7 265F (64 GB) | PyTorch, CPU (20 threads) | f32 | 5.3 | 12.7 | 16.4 | 16.4 | 16.1 | 1.7 |
| Core Ultra 7 265F (64 GB) | OpenVINO, CPU | f32 | 3.0 | 21.6 | 21.9 | 21.9 | 21.8 | 7.7 |
| Core Ultra 7 265F (64 GB) | OpenVINO, NPU | default | not compiled within 15 min | — | — | — | — | — |

## Decoder

See [`DECODER.md`](DECODER.md): the INT8-weight text-tower IR matched PyTorch
FP32 on 32 of 32 greedy tokens for the s0050 chest case (Core Ultra 7 265F, CPU).
