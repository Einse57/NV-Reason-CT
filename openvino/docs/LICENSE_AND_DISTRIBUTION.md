# License and product / distribution constraints

## Governing license

**OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)** applies to NVIDIA NV-Reason-CT model materials (weights, architecture, and related artifacts provided under that license), including derivatives.

Sources:

- https://huggingface.co/nvidia/NV-Reason-CT (card: OpenMDW-1.1)
- https://github.com/NVIDIA-Medtech/NV-Reason-CT (`LICENSE`)
- Canonical text: https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1
- Local copy: repo root `LICENSE`

Community ONNX port weights (`jarrelscy/nv-reason-ct-onnx`) are also OpenMDW-1.1 (derived). The `nv-reason-ct-web` **application** code is MIT; that does **not** re-license the model weights.

## What OpenMDW-1.1 allows (summary — read the full text)

- Permission to deal in Model Materials **without restriction**, including copyright, patent, database, and trade-secret rights embodied therein, subject to compliance.
- **Outputs** of the Model Materials have **no additional restrictions** under this agreement.
- Model Materials are provided **AS IS**, without warranty; providers have **no liability**.

## Distribution obligations (product-relevant)

If you **distribute any portion** of the Model Materials (including converted OpenVINO IR, quantized weights, or finetunes derived from them), you **must**:

1. Retain a **copy of the OpenMDW-1.1 agreement** in the distribution; and
2. Retain all applicable **copyright notices and other notices of origin**.

## Patent termination

If you file/maintain/voluntarily participate in a lawsuit asserting that the Model Materials infringe a patent or copyright (except as a response to a corresponding suit first brought against you), rights under the agreement **terminate**.

## Clearing third-party rights

You are solely responsible for clearing other persons' rights that may apply to the Model Materials or any use thereof, obtaining necessary consents, and performing due diligence.

## Medical / product use (from NVIDIA model card — not license text)

NV-Reason-CT is an **open research and development foundation**, **not** an autonomous diagnostic system or cleared clinical product. It must **not** be used for clinical diagnosis or treatment decisions without appropriate regulatory pathway, validation, and qualified medical review. Generated reasoning is reviewable model output, not guaranteed internal computation.

## This repository's artifacts

| Artifact | License posture |
|---|---|
| Scripts under `openvino/` | Same as parent repo unless noted; do not strip OpenMDW notices when shipping IR |
| Generated OpenVINO IR / compressed weights | **Derivatives of Model Materials → OpenMDW-1.1**; ship `LICENSE` + notices; do not commit multi-GB binaries to git by default |
| Tokenizer files copied from HF | Part of Model Materials → OpenMDW-1.1 |

**Do not** push multi-gigabyte `.bin` / `.safetensors` / ONNX external data to GitHub. Document download + export commands instead (see [`../README.md`](../README.md)).

## Test data used for the results in `RESULTS.md`

One CT volume from the TotalSegmentator CT dataset v2.0.1 (CC BY 4.0), used
for measurement only and not redistributed here. See `RESULTS.md` for the
citation.
