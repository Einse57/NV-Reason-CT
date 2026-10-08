# SPDX-License-Identifier: OpenMDW-1.1
"""Minimal OpenVINO agreement test for the Primus 3D encoder graph.

Uses a small randomly initialized Primus with the same flags as the
NV-Reason-CT encoder (3D rotary embeddings, scaled inner attention, identity
up-projection, LayerNorm/GELU merger), so it runs in seconds without the
checkpoint. Skipped when openvino, torch or dynamic_network_architectures is
not installed.
"""

import numpy as np
import pytest

ov = pytest.importorskip("openvino")
# The repo's openvino/ folder can shadow the package as a namespace package.
if not hasattr(ov, "convert_model"):
    pytest.skip("openvino package not installed", allow_module_level=True)
torch = pytest.importorskip("torch")
primus = pytest.importorskip("dynamic_network_architectures.architectures.primus")


class TinyEncoder(torch.nn.Module):
    def __init__(self, embed_dim=144, out_dim=64, input_shape=(32, 32, 32)):
        super().__init__()
        self.sub_vision = primus.Primus(
            input_channels=1,
            num_classes=1,
            eva_depth=2,
            eva_numheads=2,
            embed_dim=embed_dim,
            patch_embed_size=(8, 8, 8),
            input_shape=input_shape,
            use_rot_pos_emb=True,
            use_abs_pos_embed=False,
            drop_path_rate=0.2,
            init_values=0.1,
            scale_attn_inner=True,
            num_register_tokens=0,
        )
        self.sub_vision.up_projection = torch.nn.Identity()
        self.norm = torch.nn.LayerNorm(embed_dim, eps=1e-6)
        self.fc1 = torch.nn.Linear(embed_dim, embed_dim)
        self.fc2 = torch.nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        x = self.sub_vision(x).permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x.view(-1, x.shape[-1]))
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


def test_tiny_primus_openvino_cpu_matches_torch():
    torch.manual_seed(0)
    model = TinyEncoder().eval()
    x = torch.from_numpy(
        np.random.default_rng(0).uniform(-1, 1, (1, 1, 32, 32, 32)).astype(np.float32)
    )
    with torch.no_grad():
        ref = model(x).numpy()
        ov_model = ov.convert_model(model, example_input=x)
    compiled = ov.Core().compile_model(
        ov_model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"}
    )
    out = next(iter(compiled({0: x.numpy()}).values()))
    assert out.shape == ref.shape == (64, 64)
    a, b = ref.ravel().astype(np.float64), np.asarray(out).ravel().astype(np.float64)
    cosine = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cosine > 0.99999
    assert np.max(np.abs(a - b)) < 1e-3
