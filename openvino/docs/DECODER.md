# Text tower (decoder) export: status

NV-Reason-CT's language model is Qwen3.5-4B with hybrid layers (3 Gated
DeltaNet linear-attention layers per full-attention layer), wrapped in the
custom `VLM3D_ForConditionalGeneration`.

## What works

1. `extract_text_tower.py` writes the language tower as a text-only
   `qwen3_5_text` / `Qwen3_5ForCausalLM` checkpoint (427 tensors, 9.7 GB BF16).
2. `try_export_decoder_optimum.py` exports it with optimum-intel
   `text-generation-with-past` and INT8 weight compression (NNCF), giving a
   stateful IR of about 4.2 GB (inputs `input_ids`, `attention_mask`,
   `position_ids` [batch, seq], `beam_idx`; output `logits`). Export took 44 s
   on a 64 GB host.
3. `decoder_token_match.py` splices an `inputs_embeds` Parameter in place of the
   embedding Gather, feeds the same embeddings (prompt tokens plus the 13,824
   encoder tokens) to OpenVINO and to PyTorch FP32, and compares greedy output.

Versions: optimum-intel 2.2.0, optimum 2.3.0, transformers 5.2.0, OpenVINO
2026.4.1, NNCF 3.4.0, torch 2.14.1+cpu. The PyTorch reference used transformers
5.6.2 (upstream pin).

## Token match (one case)

TotalSegmentator CT s0050, chest crop, encoder tokens from the PyTorch FP32
encoder, upstream chat template with `enable_thinking=False` and the default
chest prompt; 13,844 prompt tokens; 32 greedy tokens; OpenVINO CPU with
`INFERENCE_PRECISION_HINT=f32`.

| Metric | Value |
|---|---|
| Greedy tokens identical | 32 / 32 |
| First-step logits cosine | 0.99854 |
| First-step top-1 equal / top-5 overlap | yes / 4 of 5 |

Both backends produced: `TECHNIQUE: IV contrast CT.` / `FINDINGS:` /
`Medical devices: None present.` / `CHEST:` / `Lungs and airways:`.

## Limits

- **Positions.** The text-only export takes 1-D text positions. The full
  model uses 3-D M-RoPE positions (t, h, w) for the image tokens, so the check
  above runs both backends with 1-D positions. It measures OpenVINO vs PyTorch
  agreement for the exported graph, not end-to-end report fidelity. Full report
  generation needs a position input that carries the 3-D grid, i.e. the
  `image-text-to-text` export path with a custom 3D vision tower.
- **Export path.** optimum-intel registers `qwen3_5` only for
  `image-text-to-text` (huggingface/optimum-intel#1721, open); the text-only
  route above works around that.
- **Version pins.** optimum-intel 2.2.0 with transformers 5.5.4 fails with
  `ValueError: The current version of Transformers does not allow for the
  export of the model. Maximum required is 5.2.*, got: 5.5.4`. Use
  transformers 5.2.x for export only.
- **INT4** was not tried: hybrid Qwen3.5 INT4 output is reported incoherent in
  huggingface/optimum-intel#1722.
- Only one case and 32 tokens were checked.

## Community ONNX port (negative control)

`jarrelscy/nv-reason-ct-onnx` decoder graphs do not load in OpenVINO: no
conversion rule for `com.microsoft.LinearAttention`, `LinearAttentionGate`,
`CausalConvWithState`, `GatedRMSNorm`, `MRotaryEmbedding`, plus failures on
`LpNormalization-22` and `SimplifiedLayerNormalization-22`. Full text:
[`decoder_read_model_error.txt`](decoder_read_model_error.txt).
