# OpenVINO tooling for NV-Reason-CT

Scripts to run the NV-Reason-CT 3D CT encoder (Primus ViT + merger) with
OpenVINO, check it against the PyTorch reference, and an experimental export
of the Qwen3.5 text tower with optimum-intel. Nothing here changes the default
PyTorch inference path; all dependencies are optional.

Research and development use only; not for clinical diagnosis. Model materials
and their derivatives (including OpenVINO IR) are OpenMDW-1.1; see
[`docs/LICENSE_AND_DISTRIBUTION.md`](docs/LICENSE_AND_DISTRIBUTION.md).

## Status

| Component | Status |
|---|---|
| Encoder, FP32 IR from `nvidia/NV-Reason-CT` | Exports; matches PyTorch FP32 on CPU and Intel GPU (cosine ≥ 0.99999999998) |
| Encoder on Intel NPU | Did not compile within 15 minutes on Core Ultra 9 285H or Core Ultra 7 265F |
| Text tower (Qwen3.5 hybrid) via optimum-intel, INT8 weights | Exports (experimental); 32/32 greedy tokens match PyTorch FP32 on one CT case |
| Full report generation in OpenVINO (3D M-RoPE positions) | Not implemented; see [`docs/DECODER.md`](docs/DECODER.md) |

Measured agreement and timing: [`docs/RESULTS.md`](docs/RESULTS.md).

## Setup

```bash
pip install -r requirements.txt -r openvino/requirements-openvino.txt
# local snapshot of the model (about 10.6 GB)
huggingface-cli download nvidia/NV-Reason-CT --local-dir /path/to/NV-Reason-CT
```

## Encoder

```bash
# 1. FP32 IR: volume [1,1,192,192,192] -> tokens [13824,2560]
python openvino/scripts/export_encoder_fp32.py --model-dir /path/to/NV-Reason-CT \
    --out-dir openvino/ir/encoder_fp32

# 2. Agreement vs PyTorch FP32 CPU (each device in its own process, compile capped)
python openvino/scripts/agreement_encoder.py --model-dir /path/to/NV-Reason-CT \
    --ir openvino/ir/encoder_fp32/vision_fp32.xml --devices CPU GPU NPU --gpu-large-alloc \
    --input ct.nii.gz:chest --input synthetic:0 --out agreement.json

# 3. Timing, one backend per process
python openvino/scripts/bench_encoder.py --backend torch --model-dir /path/to/NV-Reason-CT --runs 5
python openvino/scripts/bench_encoder.py --backend ov --device CPU \
    --ir openvino/ir/encoder_fp32/vision_fp32.xml --runs 5
```

`agreement_encoder.py` and `bench_encoder.py` pin `INFERENCE_PRECISION_HINT=f32`
on CPU and GPU. On Intel GPU, FP32 needs `--gpu-large-alloc`
(`GPU_ENABLE_LARGE_ALLOCATIONS=YES`): the attention scores for 13,824 tokens
(12 heads × 13,824² × 4 bytes ≈ 9.2 GB) exceed the default 4 GB allocation limit.
Inputs can be `synthetic[:seed]`, a preprocessed `.npy`, or a NIfTI volume, which
is preprocessed with the upstream `ImageLoader3D` (`:chest` or `:abdomen` crop).

## Text tower (experimental)

Use a separate environment (`openvino/requirements-decoder-export.txt`):
optimum-intel 2.2.0 only exports `qwen3_5` with transformers 5.2.x, while
upstream inference pins transformers 5.6.2.

```bash
python openvino/scripts/extract_text_tower.py --model-dir /path/to/NV-Reason-CT --out-dir hf_lm_only
python openvino/scripts/try_export_decoder_optimum.py -m hf_lm_only \
    --out openvino/ir/decoder_int8 --weight-format int8
```

`decoder_token_match.py` compares greedy decoding of the exported text tower
against PyTorch on the same `inputs_embeds` (CT encoder tokens spliced in). See
[`docs/DECODER.md`](docs/DECODER.md) for what it does and does not cover.

## Tests

```bash
pytest openvino/tests   # skipped when openvino or torch is not installed
```

## Artifacts

IR, weights and intermediate arrays are not committed (`.gitignore`). The FP32
encoder IR is about 597 MB; the INT8 text-tower IR is about 4.2 GB.
