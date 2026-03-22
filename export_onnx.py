#!/usr/bin/env python3
"""Export Opla model to ONNX for CPU-only deployment.

Exports the dual-BERT + heads architecture to ONNX format, enabling
inference with onnxruntime instead of requiring PyTorch (~50MB vs ~2GB).

For AG/med models (single shared BERT): exports one combined ONNX model.
For MG models (dual BERT): exports two ONNX models (pos + dp).

Usage:
    # Export AG model
    python export_onnx.py --lang grc --weights weights/grc/opla_grc.pt

    # Export MG model
    python export_onnx.py --lang el --weights weights/el/opla_el.pt

    # Custom output directory
    python export_onnx.py --lang grc --output weights/grc/onnx/
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from opla.model import OplaModel
from opla.labels import pos_labels, dp_labels

BERT_MODELS = {
    "el": "nlpaueb/bert-base-greek-uncased-v1",
    "grc": "pranaydeeps/Ancient-Greek-BERT",
    "med": "pranaydeeps/Ancient-Greek-BERT",
}


class PosWrapper(torch.nn.Module):
    """Wrapper that runs POS BERT + heads and returns stacked logits."""

    def __init__(self, model):
        super().__init__()
        self.bert = model.pos_bert
        self.heads = model.pos_heads
        self.feat_names = sorted(model.pos_heads.keys())

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Stack all feature logits into one tensor for ONNX compatibility
        # Output shape: (num_feats, batch, seq_len, max_labels)
        # Each feature may have different label count, so we pad to max
        logits = []
        for feat in self.feat_names:
            logits.append(self.heads[feat](out))
        return logits


class DpWrapper(torch.nn.Module):
    """Wrapper that runs DP BERT + biaffine heads."""

    def __init__(self, model):
        super().__init__()
        self.bert = model.dp_bert
        self.shared = model.shared_bert
        self.arc_head = model.arc_head
        self.arc_dep = model.arc_dep
        self.rel_head = model.rel_head
        self.rel_dep = model.rel_dep
        self.arc_bias = model.arc_bias
        self.w_arc = model.w_arc
        self.u_rel = model.u_rel
        self.w_rel_head = model.w_rel_head
        self.w_rel_dep = model.w_rel_dep
        self.rel_bias = model.rel_bias
        self.deprel_linear_2 = model.deprel_linear_2
        self.numrels = model.numrels
        self.relu = model.relu

    def forward(self, input_ids, attention_mask):
        dp_out = self.bert(input_ids, attention_mask=attention_mask)[0]
        bs, mseq = dp_out.shape[:2]

        arc_h = self.relu(self.arc_head(dp_out))
        arc_d = self.relu(self.arc_dep(dp_out))
        rel_h = self.relu(self.rel_head(dp_out))
        rel_d = self.relu(self.rel_dep(dp_out))

        arc_scores = (
            arc_h @ (arc_d @ self.w_arc).transpose(1, 2)
            + arc_h @ self.arc_bias
        )

        label_biaffine = rel_d @ self.u_rel
        label_biaffine = label_biaffine.reshape(bs, mseq, self.numrels, 768)
        label_biaffine = label_biaffine @ rel_h.transpose(1, 2).unsqueeze(1)
        label_biaffine = label_biaffine.transpose(2, 3)

        label_head_affine = rel_h.unsqueeze(2) @ self.w_rel_head
        label_dep_affine = rel_d.unsqueeze(2) @ self.w_rel_dep

        rel_scores = (
            label_biaffine + label_head_affine + label_dep_affine
            + self.rel_bias
        ).reshape(bs, mseq, mseq, self.numrels)

        return arc_scores, rel_scores


class JointWrapper(torch.nn.Module):
    """Wrapper for shared-BERT models (AG/med): one pass for POS + DP."""

    def __init__(self, model):
        super().__init__()
        self.bert = model.pos_bert
        self.heads = model.pos_heads
        self.feat_names = sorted(model.pos_heads.keys())
        self.arc_head = model.arc_head
        self.arc_dep = model.arc_dep
        self.rel_head = model.rel_head
        self.rel_dep = model.rel_dep
        self.arc_bias = model.arc_bias
        self.w_arc = model.w_arc
        self.u_rel = model.u_rel
        self.w_rel_head = model.w_rel_head
        self.w_rel_dep = model.w_rel_dep
        self.rel_bias = model.rel_bias
        self.numrels = model.numrels
        self.relu = model.relu

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask)[0]
        bs, mseq = out.shape[:2]

        # POS logits
        pos_logits = []
        for feat in self.feat_names:
            pos_logits.append(self.heads[feat](out))

        # DP scores (same hidden states)
        arc_h = self.relu(self.arc_head(out))
        arc_d = self.relu(self.arc_dep(out))
        rel_h = self.relu(self.rel_head(out))
        rel_d = self.relu(self.rel_dep(out))

        arc_scores = (
            arc_h @ (arc_d @ self.w_arc).transpose(1, 2)
            + arc_h @ self.arc_bias
        )

        label_biaffine = rel_d @ self.u_rel
        label_biaffine = label_biaffine.reshape(bs, mseq, self.numrels, 768)
        label_biaffine = label_biaffine @ rel_h.transpose(1, 2).unsqueeze(1)
        label_biaffine = label_biaffine.transpose(2, 3)

        label_head_affine = rel_h.unsqueeze(2) @ self.w_rel_head
        label_dep_affine = rel_d.unsqueeze(2) @ self.w_rel_dep

        rel_scores = (
            label_biaffine + label_head_affine + label_dep_affine
            + self.rel_bias
        ).reshape(bs, mseq, mseq, self.numrels)

        return (*pos_logits, arc_scores, rel_scores)


def export(lang: str, weights_path: Path, output_dir: Path):
    """Export model to ONNX."""
    bert_name = BERT_MODELS[lang]
    print(f"Loading BERT: {bert_name}")
    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    # Load checkpoint
    print(f"Loading weights: {weights_path}")
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)

    feat_sizes = ckpt.get("feat_sizes",
                          {k: len(v) for k, v in pos_labels.items()})
    num_deprels = ckpt.get("num_deprels", len(dp_labels))
    shared = ckpt.get("lang", lang) in ("grc", "med")

    if shared:
        bert = AutoModel.from_pretrained(bert_name)
        model = OplaModel(bert, feat_sizes=feat_sizes, num_deprels=num_deprels)
    else:
        pos_bert = AutoModel.from_pretrained(bert_name)
        dp_bert = AutoModel.from_pretrained(bert_name)
        model = OplaModel(pos_bert, dp_bert, feat_sizes=feat_sizes,
                          num_deprels=num_deprels)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Create dummy input
    dummy_text = "ὁ Σωκράτης ἔφη"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length",
                       max_length=64, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output_dir.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
    }

    if shared:
        # Single ONNX model for AG/med
        wrapper = JointWrapper(model)
        wrapper.eval()
        feat_names = sorted(model.pos_heads.keys())

        out_names = [f"pos_{f}" for f in feat_names] + ["arc_scores", "rel_scores"]
        out_path = output_dir / "opla_joint.onnx"

        print(f"Exporting joint model to {out_path}...")
        torch.onnx.export(
            wrapper, (input_ids, attention_mask),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=out_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"  {out_path}: {out_path.stat().st_size / 1e6:.1f} MB")
    else:
        # Separate POS and DP models for MG
        pos_wrapper = PosWrapper(model)
        pos_wrapper.eval()
        feat_names = sorted(model.pos_heads.keys())

        pos_path = output_dir / "opla_pos.onnx"
        print(f"Exporting POS model to {pos_path}...")
        torch.onnx.export(
            pos_wrapper, (input_ids, attention_mask),
            str(pos_path),
            input_names=["input_ids", "attention_mask"],
            output_names=[f"pos_{f}" for f in feat_names],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"  {pos_path}: {pos_path.stat().st_size / 1e6:.1f} MB")

        dp_wrapper = DpWrapper(model)
        dp_wrapper.eval()

        dp_path = output_dir / "opla_dp.onnx"
        print(f"Exporting DP model to {dp_path}...")
        torch.onnx.export(
            dp_wrapper, (input_ids, attention_mask),
            str(dp_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["arc_scores", "rel_scores"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"  {dp_path}: {dp_path.stat().st_size / 1e6:.1f} MB")

    # Save metadata for inference
    import json
    meta = {
        "lang": lang,
        "bert_model": bert_name,
        "shared_bert": shared,
        "feat_names": sorted(model.pos_heads.keys()),
        "feat_sizes": feat_sizes,
        "num_deprels": num_deprels,
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Export Opla model to ONNX")
    parser.add_argument("--lang", required=True, choices=["el", "grc", "med"])
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: same as weights)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.weights.parent / "onnx"

    export(args.lang, args.weights, args.output)


if __name__ == "__main__":
    main()
