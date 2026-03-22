"""ONNX inference backend for Opla.

Replaces PyTorch model with onnxruntime for CPU-only deployment.
Requires: pip install onnxruntime (~50MB vs ~2GB for torch+transformers)
"""

import json
import numpy as np
from pathlib import Path


class OplaONNX:
    """ONNX-backed model that mimics OplaModel's forward() output."""

    def __init__(self, onnx_dir: Path):
        import onnxruntime as ort

        meta_path = onnx_dir / "meta.json"
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.feat_names = self.meta["feat_names"]

        # Find the ONNX model file
        joint_path = onnx_dir / "opla_joint.onnx"
        if joint_path.exists():
            self.session = ort.InferenceSession(
                str(joint_path),
                providers=["CPUExecutionProvider"],
            )
        else:
            raise FileNotFoundError(f"No ONNX model found in {onnx_dir}")

        # Map output names
        self._output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, input_ids, attention_mask):
        """Run inference, returning outputs matching OplaModel.forward().

        Args:
            input_ids: numpy array (batch, seq_len) or torch.Tensor
            attention_mask: numpy array (batch, seq_len) or torch.Tensor

        Returns:
            pos_logits: dict {feat_name: torch.Tensor (batch, seq, n_labels)}
            arc_scores: torch.Tensor (batch, seq, seq)
            rel_scores: torch.Tensor (batch, seq, seq, numrels)
        """
        import torch

        # Convert to numpy if needed
        if hasattr(input_ids, "numpy"):
            input_ids_np = input_ids.cpu().numpy()
            attention_mask_np = attention_mask.cpu().numpy()
        else:
            input_ids_np = input_ids
            attention_mask_np = attention_mask

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids_np.astype(np.int64),
                "attention_mask": attention_mask_np.astype(np.int64),
            },
        )

        # Map outputs back to the expected format
        pos_logits = {}
        arc_scores = None
        rel_scores = None

        for name, arr in zip(self._output_names, outputs):
            tensor = torch.from_numpy(arr)
            if name.startswith("pos_"):
                feat = name[4:]  # strip "pos_" prefix
                pos_logits[feat] = tensor
            elif name == "arc_scores":
                arc_scores = tensor
            elif name == "rel_scores":
                rel_scores = tensor

        return pos_logits, arc_scores, rel_scores

    def to(self, device):
        """No-op for ONNX (always runs on CPU)."""
        return self

    def eval(self):
        """No-op for ONNX (always in eval mode)."""
        return self
