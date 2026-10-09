# Encoder and decoder results

Unless a section says otherwise, numbers were measured on 2026-10-07 with
OpenVINO 2026.4.1, torch 2.14.1+cpu and transformers 5.6.2 (decoder export:
5.2.0) on Windows 11. The PyTorch XPU section was measured on 2026-10-08.
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

### OpenVINO CPU threads

On the Core Ultra 9 285H, the CPU plugin's default `LATENCY` hint runs on the
6 P-cores (`INFERENCE_NUM_THREADS` = 6), while PyTorch uses 16 threads. With
`INFERENCE_NUM_THREADS=14` (`bench_encoder.py --ov-threads 14`; the plugin uses
at most 14 threads on this CPU) the encoder runs in 21.3 s instead of 26.8 s in
the 2026-10-08 session below. Peak RSS (8.2–8.3 GiB) does not depend on the
thread count. The input shape is static, and precision is pinned to f32 on CPU.

## PyTorch XPU and OpenVINO GPU f16 (2026-10-08)

Same host (Core Ultra 9 285H, Arc 140T iGPU), input (CT s0050 chest crop) and
method as above, using `bench_encoder.py` with torch 2.14.1+xpu and OpenVINO
2026.4.1. All rows ran in one interactive desktop session with no other GPU
load. The reference for agreement is the PyTorch FP32 CPU output from the
agreement run. For `torch` rows, the whole encoder and the input are cast to
the listed dtype. With `torch.compile`, the first run includes compilation. On
Windows this needs MSVC (Visual Studio 2022 Build Tools, C++ workload) with
`vcvars64.bat` loaded.

| Backend / device | Precision | Load (s) | First run (s) | p50 (s) | Min (s) | Peak RSS (GiB) | 1 − cosine | 1 − min per-token cosine | Max abs diff | Rel. L2 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch, CPU (16 threads) | f32 | 5.7 | 22.4 | 22.38 | 22.32 | 2.8 | 9.7e-14 | 1.8e-14 | 0 | 0 |
| PyTorch, XPU (Arc 140T iGPU) | fp16 | 6.4 | 2.9 | 2.55 | 2.54 | 3.6 | 1.3e-06 | 7.6e-06 | 4.1e-03 | 1.6e-03 |
| PyTorch, XPU (Arc 140T iGPU) | bf16 | 5.4 | 2.9 | 2.52 | 2.50 | 3.6 | 7.8e-05 | 6.4e-04 | 2.8e-02 | 1.2e-02 |
| PyTorch, XPU, `torch.compile` | fp16 | 6.0 | 56.9 (incl. compile) | 2.57 | 2.53 | 3.5 | 1.2e-06 | 8.0e-06 | 3.9e-03 | 1.5e-03 |
| PyTorch, XPU, `torch.compile` | bf16 | 7.0 | 53.2 (incl. compile) | 2.57 | 2.50 | 3.5 | 7.4e-05 | 8.4e-04 | 3.0e-02 | 1.2e-02 |
| OpenVINO, CPU (default, 6 threads) | f32 | 2.8 | 26.8 | 26.81 | 26.75 | 8.2 | 6.7e-13 | 5.6e-12 | 2.9e-06 | 1.1e-06 |
| OpenVINO, CPU (`INFERENCE_NUM_THREADS=14`) | f32 | 3.0 | 21.2 | 21.32 | 21.21 | 8.3 | 6.7e-13 | 5.6e-12 | 2.9e-06 | 1.1e-06 |
| OpenVINO, Arc 140T iGPU | f16 | 2.1 | 2.4 | 2.45 | 2.45 | 3.4 | 1.7e-06 | 2.3e-05 | 5.1e-03 | 1.9e-03 |
| OpenVINO, Arc 140T iGPU (large alloc) | f32 | 5.9 | 18.8 | 18.80 | 18.78 | 20.4 | 1.1e-11 | 4.2e-10 | 2.7e-05 | 4.6e-06 |

Notes:

- PyTorch XPU in f32 is not usable for this encoder on this iGPU. The f32
  attention materializes one 8.54 GiB score tensor, and allocation fails above
  the 4 GiB per-allocation limit. With `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`
  it runs, but the output does not match the reference (rel. L2 1.15). Use fp16
  or bf16 on XPU. Those use a fused attention kernel (peak XPU allocation about
  0.7 GiB).
- `torch.compile` fails on this encoder under `torch.inference_mode()` with a
  Dynamo guard error (`'___from_numpy(self.drop_prob)' dispatch key set
  mismatch`). Run the compiled module under `torch.no_grad()` instead, which
  `bench_encoder.py` does. Warm time is the same as eager on this host.
- The OpenVINO GPU f32 row repeats the 2026-10-07 configuration in this
  session.
- Raw results: [`../results/encoder_timing_20261008.json`](../results/encoder_timing_20261008.json).

## Decoder

See [`DECODER.md`](DECODER.md): the INT8-weight text-tower IR matched PyTorch
FP32 on 32 of 32 greedy tokens for the s0050 chest case (Core Ultra 7 265F, CPU).
