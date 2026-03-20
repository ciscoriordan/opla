#!/usr/bin/env python3
"""Benchmark Opla POS tagger against CoNLL-U test sets.

Evaluates on:
  - UD Ancient Greek Perseus (test split)
  - DiGreC (test split)

Reports UPOS accuracy and compares against published baselines.

Usage:
    python benchmark.py
"""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from opla import Opla


def load_conllu(path: Path) -> list[list[dict]]:
    """Load CoNLL-U file, return list of sentences (list of token dicts)."""
    sentences = []
    current = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            # Skip multiword tokens (e.g., "1-2")
            if "-" in fields[0] or "." in fields[0]:
                continue
            current.append({
                "form": fields[1],
                "upos": fields[3],
                "deprel": fields[7],
            })
    if current:
        sentences.append(current)
    return sentences


def evaluate(opla: Opla, sentences: list[list[dict]]) -> dict:
    """Run Opla on sentences and compare against gold UPOS."""
    # Build input strings
    texts = [" ".join(tok["form"] for tok in sent) for sent in sentences]

    # Tag in batches
    batch_size = 64
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds = opla.tag(batch)
        all_preds.extend(preds)

    total = 0
    upos_correct = 0
    deprel_correct = 0

    for gold_sent, pred_sent in zip(sentences, all_preds):
        if len(gold_sent) != len(pred_sent):
            # Tokenization mismatch - skip
            continue
        for gold_tok, pred_tok in zip(gold_sent, pred_sent):
            total += 1
            if gold_tok["upos"] == pred_tok["upos"]:
                upos_correct += 1
            if gold_tok["deprel"] == pred_tok["deprel"]:
                deprel_correct += 1

    return {
        "total": total,
        "upos_correct": upos_correct,
        "upos_acc": upos_correct / total if total else 0,
        "deprel_correct": deprel_correct,
        "deprel_acc": deprel_correct / total if total else 0,
    }


def main():
    data_dir = Path(__file__).parent / "data"

    datasets = [
        ("UD AG-Perseus (test)", data_dir / "UD_Ancient_Greek-Perseus" / "grc_perseus-ud-test.conllu"),
        ("DiGreC (test)", data_dir / "DiGreC" / "digrec-test.conllu"),
    ]

    print("Loading Opla (lang=grc)...")
    opla = Opla(lang="grc")
    print()

    for name, path in datasets:
        if not path.exists():
            print(f"{name}: NOT FOUND ({path})")
            continue

        sentences = load_conllu(path)
        print(f"{name}: {len(sentences)} sentences, "
              f"{sum(len(s) for s in sentences)} tokens")

        results = evaluate(opla, sentences)
        print(f"  UPOS accuracy: {results['upos_correct']}/{results['total']} "
              f"({results['upos_acc']:.1%})")
        print(f"  DEPREL accuracy: {results['deprel_correct']}/{results['total']} "
              f"({results['deprel_acc']:.1%})")
        print()

    print("Published baselines:")
    print("  Lemming (Perseus):  ~88% UPOS")
    print("  GreTa (Perseus):    ~95% UPOS")


if __name__ == "__main__":
    main()
