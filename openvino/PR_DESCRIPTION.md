# OpenVINO CPU path for NV-Reason-CT (encoder + decoder attempt)

Draft PR on **Einse57/NV-Reason-CT** only (does not target upstream `NVIDIA-Medtech/NV-Reason-CT`).

## Summary

Adds `openvino/` tooling to export / run the **Primus 3D CT encoder** on OpenVINO CPU, documents **OpenMDW-1.1** distribution constraints, and records an honest **decoder blocker** for the hybrid Qwen3.5-4B linear-attention stack.

This is **not** a packaging of the community WebGPU/ORT-genai ONNX port as the product path; that port is used only as a validated intermediate for the encoder IR and as a negative control for the decoder.

## Licenses / product constraints

See [`openvino/docs/LICENSE_AND_DISTRIBUTION.md`](docs/LICENSE_AND_DISTRIBUTION.md).

- **Model materials:** OpenMDW-1.1 (same as `nvidia/NV-Reason-CT` and this repo’s `LICENSE`).
- **Distribution:** retain OpenMDW-1.1 text + copyright/origin notices when shipping IR / quantized derivatives.
- **Outputs:** no extra OpenMDW restrictions on model outputs.
- **AS IS / no liability;** patent-assertion termination clause applies.
- **Not a medical device:** research/developer foundation only; not for clinical diagnosis/treatment decisions.

## What exported

| Artifact | Status |
|---|---|
| Encoder OpenVINO IR (`volume` `[1,1,192,192,192]` → `tokens` `[13824,2560]`) | **OK** — via `ov.convert_model` on community int8 ONNX derived from upstream |
| Encoder FP32 IR directly from `nvidia/NV-Reason-CT` safetensors | **Soft-blocked** on 15 GiB box (OOM during ONNX/OV tracing); script provided for ≥32 GiB hosts |
| Decoder OpenVINO IR / OVGenAI | **Blocked** — see [`BLOCKER.md`](BLOCKER.md) |
| NNCF int8/int4 on decoder | **Skipped** (no working FP decoder export) |

Large weight binaries are **gitignored**; reproduce with scripts + documented downloads.

## Agreement

- Upstream FP32 torch encoder (zero volume) vs community-int8→OV cosine ≈ **0.99972**
- Decoder: N/A (does not load)

## Box benches (labeled **box-only / non-target**)

Encoder-only OpenVINO CPU (random ±0.1 volume, 3 timed runs after 1 warmup):

- mean latency ≈ **12.0 s** / forward
- peak RSS ≈ **7.1 GiB**

**Published baseline (not comparable):** RTX PRO 6000 ≈ **25 s/report** for full NV-Reason-CT VLM on GPU.

## Fork / branch

- Fork: https://github.com/Einse57/NV-Reason-CT
- Branch: `openvino-cpu`
- Draft PR: this PR (fork main ← openvino-cpu)

## Blockers (decoder)

1. Community ONNX decoder: OpenVINO cannot convert `com.microsoft.LinearAttention`, `LinearAttentionGate`, `CausalConvWithState`, `GatedRMSNorm`, `MRotaryEmbedding` (+ LN/LpNormalization issues).
2. optimum-intel (git main) + transformers 5.6.2 (NV pin): `ImportError: Qwen3_5DynamicCache` during Qwen3.5 export patcher.
3. Custom `VLM3D_ForConditionalGeneration` + 10.6 GB BF16 checkpoint exceed comfortable 15 GiB export headroom.

**Honest assessment:** hybrid linear-attention layers are **not** yet cleanly exportable for this model on the pinned stack; encoder OV path is useful standalone.
