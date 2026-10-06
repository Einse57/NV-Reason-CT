# Blockers — NV-Reason-CT OpenVINO export (2026-10-06)

## Decoder: BLOCKED for OpenVINO (community ONNX and hybrid layers)

### A. Community ONNX decoder (`jarrelscy/nv-reason-ct-onnx` int4/int8)

`ov.Core().read_model(decoder.onnx)` fails. Exact failure class:

1. **No conversion rule** for Microsoft custom ops used by the hybrid Qwen3.5 stack:
   - `com.microsoft.LinearAttention`
   - `com.microsoft.LinearAttentionGate`
   - `com.microsoft.CausalConvWithState`
   - `com.microsoft.GatedRMSNorm`
   - `com.microsoft.MRotaryEmbedding`
2. **Failed conversions** for:
   - `LpNormalization-22` ("Element not supported")
   - `SimplifiedLayerNormalization-22` (dtype mismatch, dynamic vs f32)
   - `com.microsoft.SkipSimplifiedLayerNormalization` (dtype mismatch)

Full text: `docs/decoder_read_model_error.txt`.

This path is the **WebGPU / ORT-genai ONNX port**, not an optimum-intel / OVGenAI IR. It is retained only as a negative control.

### B. Upstream PyTorch → optimum-intel / OVGenAI decoder

NV-Reason-CT wraps **Qwen3.5-4B** (`model_type=qwen3_5`, hybrid 3:1 GatedDeltaNet linear-attention + full attention) plus a custom Primus 3D ViT. Challenges on this box:

| Issue | Detail |
|---|---|
| Weight size | `model.safetensors` ≈ **10.6 GB BF16**; language_model alone ≈ **8.4 GB** (+ lm_head ≈ 1.3 GB). Box RAM ≈ **15 GiB**. |
| Custom VLM | `VLM3D_ForConditionalGeneration` is not a stock Qwen3.5 VLM; 3D Primus tower is out of band for `OVModelForVisualCausalLM`. |
| Hybrid layers | OVGenAI documents Qwen3.5 support (needs recent optimum-intel + transformers≈5.2+), but **NV-Reason-CT finetuned weights** still need a text-only export of the language tower. |
| FP32 encoder export OOM | `torch.onnx.export` / `ov.convert_model` of upstream Primus FP32 **OOM-killed (exit 137)** on this 15 GiB box during graph tracing. |

**Status:** Decoder OpenVINO IR **not produced** in this pass. Encoder OV IR **is** produced (via community int8 ONNX → OV), validated against upstream PyTorch (cosine ≈ 0.9997 on zero volume).

### C. What would unblock the decoder

1. Machine with **≥32–64 GiB RAM** (or GPU) for optimum-intel export of the extracted `hf_lm_only/` Qwen3.5 text checkpoint (`weight-format int4` recommended).
2. optimum-intel / openvino-genai versions that fully support Qwen3.5 hybrid cache (see OVGenAI supported-models notes; transformers pin per docs).
3. Optional: mixed-precision NNCF that keeps GatedDeltaNet layers at int8/fp16 (known INT4 quality risk on larger Qwen3.5 — see huggingface/optimum-intel#1722).

## Encoder FP32 upstream IR: soft blocker (RAM)

Upstream Primus loads and runs in PyTorch FP32 on this box (~294 MB weights). Converting FP32→ONNX/OV OOMs. Delivered encoder IR is **int8 community ONNX converted with `ov.convert_model`**, with agreement vs upstream torch documented below.


### D. optimum-intel ↔ transformers version skew (Qwen3.5 probe, 2026-10-06)

Attempted `optimum.exporters.openvino.main_export` on `Qwen/Qwen3.5-0.8B` with
`task=image-text-to-text`, `weight_format=int4`, optimum-intel **git main**
(`2.3.0.dev0`), transformers **5.6.2** (NV-Reason-CT pin):

```
ImportError: cannot import name 'Qwen3_5DynamicCache'
from 'transformers.models.qwen3_5.modeling_qwen3_5'
```

Also: `task=text-generation-with-past` is rejected for `model_type=qwen3_5`
(only `image-text-to-text` registered). Stock Qwen3.5 VLM export is therefore
**not working** on this pin combination; NV-Reason-CT's custom
`VLM3D_ForConditionalGeneration` is an additional unsupported surface.

**Honesty check:** optimum-intel / OVGenAI are moving toward Qwen3.5 hybrid
support, but **this hybrid linear-attention stack is not yet reliably
exportable end-to-end** with the NV-Reason-CT-required transformers 5.6.2 pin
on this box. Do not claim a working OVGenAI decoder path yet.
