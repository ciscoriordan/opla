#!/usr/bin/env python3
"""Verify ONNX inference produces identical output to PyTorch inference.

Loads both backends, runs the same sentences through each, and compares
UPOS, head, deprel, and feats for every token. Uses AG dev data from
UD Perseus + PROIEL treebanks.
"""

import sys
from pathlib import Path
from collections import Counter

# Extract sentences from CoNLL-U dev files
def extract_sentences(conllu_path: Path, max_sentences: int = 100) -> list[str]:
    """Extract raw text from # text = lines in a CoNLL-U file."""
    sentences = []
    with open(conllu_path) as f:
        for line in f:
            if line.startswith("# text = "):
                text = line[len("# text = "):].strip()
                # Skip very long sentences (>100 words) to keep runtime sane
                if len(text.split()) <= 100:
                    sentences.append(text)
                if len(sentences) >= max_sentences:
                    break
    return sentences


def compare_results(pt_results, onnx_results, sentences):
    """Compare PyTorch vs ONNX results token by token.

    Returns (total_tokens, mismatches) where mismatches is a list of
    dicts describing each divergence.
    """
    total_tokens = 0
    mismatches = []
    fields = ["upos", "head", "deprel", "feats"]

    for sent_idx, (pt_sent, onnx_sent, text) in enumerate(
            zip(pt_results, onnx_results, sentences)):
        if len(pt_sent) != len(onnx_sent):
            mismatches.append({
                "sent_idx": sent_idx,
                "text": text[:80],
                "type": "token_count",
                "pt": len(pt_sent),
                "onnx": len(onnx_sent),
            })
            continue

        for tok_idx, (pt_tok, onnx_tok) in enumerate(zip(pt_sent, onnx_sent)):
            total_tokens += 1
            for field in fields:
                pt_val = pt_tok.get(field)
                onnx_val = onnx_tok.get(field)
                if pt_val != onnx_val:
                    mismatches.append({
                        "sent_idx": sent_idx,
                        "tok_idx": tok_idx,
                        "form": pt_tok["form"],
                        "field": field,
                        "pt": pt_val,
                        "onnx": onnx_val,
                        "text": text[:80],
                    })

    return total_tokens, mismatches


def main():
    # Collect sentences from both AG dev files
    data_dir = Path(__file__).parent / "data"
    perseus_dev = data_dir / "UD_Ancient_Greek-Perseus" / "grc_perseus-ud-dev.conllu"
    proiel_dev = data_dir / "UD_Ancient_Greek-PROIEL" / "grc_proiel-ud-dev.conllu"

    sentences = []
    for path in [perseus_dev, proiel_dev]:
        if path.exists():
            sents = extract_sentences(path, max_sentences=75)
            sentences.extend(sents)
            print(f"  {path.name}: {len(sents)} sentences")

    print(f"Total test sentences: {len(sentences)}")
    print()

    # Load PyTorch model
    print("Loading PyTorch model...")
    from opla import Opla
    pt_model = Opla(lang="grc", device="cpu", lemmatize=False)
    print("  PyTorch model loaded.")

    # Run PyTorch inference
    print("Running PyTorch inference...")
    pt_results = pt_model.tag(sentences)
    pt_tokens = sum(len(s) for s in pt_results)
    print(f"  {pt_tokens} tokens tagged.")

    # Clear the PyTorch model to save memory before loading ONNX
    del pt_model

    # Load ONNX model
    print("Loading ONNX model...")
    onnx_model = Opla(lang="grc", checkpoint="onnx", device="cpu", lemmatize=False)
    print(f"  ONNX model loaded (using_onnx={getattr(onnx_model, '_using_onnx', 'N/A')}).")

    # Run ONNX inference
    print("Running ONNX inference...")
    onnx_results = onnx_model.tag(sentences)
    onnx_tokens = sum(len(s) for s in onnx_results)
    print(f"  {onnx_tokens} tokens tagged.")
    print()

    # Compare
    total_tokens, mismatches = compare_results(pt_results, onnx_results, sentences)
    print(f"{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Total tokens compared: {total_tokens}")
    print(f"Total mismatches:      {len(mismatches)}")

    if not mismatches:
        print()
        print("PERFECT MATCH: ONNX output is identical to PyTorch.")
        return

    # Analyze mismatches by field
    field_counts = Counter(m["field"] for m in mismatches if "field" in m)
    print()
    print("Mismatches by field:")
    for field, count in field_counts.most_common():
        pct = count / total_tokens * 100
        print(f"  {field:8s}: {count:4d} ({pct:.2f}%)")

    # Show first few mismatches in detail
    print()
    detail_limit = 20
    print(f"First {min(detail_limit, len(mismatches))} mismatches:")
    print("-" * 80)
    for m in mismatches[:detail_limit]:
        if m.get("type") == "token_count":
            print(f"  Sent {m['sent_idx']}: token count differs "
                  f"(PT={m['pt']}, ONNX={m['onnx']})")
            print(f"    text: {m['text']}")
        else:
            print(f"  Sent {m['sent_idx']}, tok {m['tok_idx']} "
                  f"'{m['form']}': {m['field']} "
                  f"PT={m['pt']} vs ONNX={m['onnx']}")

    # Assess severity: classify as trivial (floating-point margin cases)
    # vs significant (clearly wrong POS, etc.)
    if "head" in field_counts or "deprel" in field_counts:
        head_mismatches = [m for m in mismatches if m.get("field") == "head"]
        if head_mismatches:
            print()
            print("NOTE: Head assignment differences are often caused by")
            print("floating-point rounding in biaffine scores. Check if the")
            print("arc score margins are small for these tokens.")

    # Summary verdict
    print()
    mismatch_rate = len(mismatches) / total_tokens * 100 if total_tokens else 0
    if mismatch_rate < 0.1:
        print(f"VERDICT: Near-identical ({mismatch_rate:.3f}% mismatch rate).")
        print("Differences are likely floating-point rounding artifacts.")
    elif mismatch_rate < 1.0:
        print(f"VERDICT: Minor divergences ({mismatch_rate:.2f}% mismatch rate).")
        print("Investigate whether these affect downstream tasks.")
    else:
        print(f"VERDICT: SIGNIFICANT divergences ({mismatch_rate:.1f}% mismatch rate).")
        print("The ONNX export may have a bug.")

    sys.exit(1 if mismatch_rate >= 1.0 else 0)


if __name__ == "__main__":
    main()
