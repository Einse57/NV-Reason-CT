# OpenVINO export notes for NV-Reason-CT

**Branch purpose:** CPU / OpenVINO paths for the Primus 3D CT encoder and (when feasible) the Qwen3.5-4B hybrid decoder, via **optimum-intel / OpenVINO GenAI / NNCF** — **not** the community WebGPU/ORT-genai ONNX port as the primary target.

## License

Model materials: **OpenMDW-1.1**. See [`docs/LICENSE_AND_DISTRIBUTION.md`](docs/LICENSE_AND_DISTRIBUTION.md) and repo root `LICENSE`. Retain license + copyright notices when distributing IR or quantized weights.

**Not for clinical diagnosis.** Research / developer foundation only.

## Status (2026-10-06 box)

| Component | OpenVINO status | Notes |
|---|---|---|
| 3D encoder (Primus + merger) | **Works (CPU)** | IR from community int8 ONNX via `ov.convert_model`; cosine ≈ **0.9997** vs upstream FP32 torch (zero volume) |
| Decoder (Qwen3.5 hybrid) | **Blocked** | Community ONNX: missing `com.microsoft.LinearAttention*` etc. Upstream optimum export: custom VLM3D + RAM; see [`BLOCKER.md`](BLOCKER.md) |
| Full report E2E | Not on this box | Published baseline: **RTX PRO 6000 ~25 s/report** (GPU). Box benches are **non-target / encoder-only**. |

## One-command encoder bench (box-only / non-target)

```bash
python openvino/scripts/bench_encoder_ov_cpu.py \
  --model path/to/vision.xml --runs 5 --out openvino/benches/encoder_ov_cpu.json
```

Labels every run as **box-only / non-target**. Do not compare encoder-only CPU ms to the 25 s/report GPU baseline.

## Reproduce encoder IR

```bash
# 1) Download community encoder ONNX (OpenMDW-1.1 derived) OR export FP32 on a large-RAM host
# 2) Convert
python openvino/scripts/export_encoder_ov.py --onnx /path/to/vision.onnx --out-dir openvino/ir/encoder
# 3) Smoke
python openvino/scripts/smoke_encoder_ov.py --model openvino/ir/encoder/vision.xml
```

Upstream FP32 export helper (needs ≥32 GiB RAM typically):

```bash
# After downloading nvidia/NV-Reason-CT model.safetensors into hf_src/
python openvino/scripts/export_encoder_minimal.py
```

## Decoder attempt (large RAM)

```bash
# Extract language_model.* + lm_head from HF safetensors into hf_lm_only/ (model_type=qwen3_5_text)
python openvino/scripts/try_export_decoder_optimum.py \
  -m ./hf_lm_only --out openvino/ir/decoder_int4 --weight-format int4 \
  --task text-generation-with-past   # or image-text-to-text for full qwen3_5 VLM
```

Honest expectation: hybrid GatedDeltaNet layers need current OVGenAI hybrid-cache support; INT4 quality on larger Qwen3.5 hybrids can degrade (optimum-intel#1722). Prefer int8 or mixed precision for GatedDeltaNet if incoherent.

## Do not commit

Multi-GB `.bin` / `.safetensors` / ONNX external data — gitignored. Document download + export instead.
