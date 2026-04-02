#!/usr/bin/env python3
"""Train Opla POS+DP heads on Universal Dependencies treebanks.

Usage:
    python train.py                          # train AG heads on Perseus + PROIEL
    python train.py --lang el --data ...     # train MG heads on custom data
    python train.py --epochs 5 --lr 3e-5     # custom hyperparameters
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from opla.labels import pos_labels, dp_labels, pos_properties
from opla.model import OplaModel
from opla.tokenize import strip_accents_and_lowercase

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


# --- CoNLL-U parsing ---

def parse_conllu(path):
    """Parse a CoNLL-U file into a list of sentences.

    Each sentence is a list of (form, upos, feats_dict, head, deprel) tuples.
    """
    sentences = []
    current = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) < 10:
                continue
            # Skip multi-word and empty tokens
            if "-" in parts[0] or "." in parts[0]:
                continue

            form = parts[1]
            upos = parts[3]
            raw_feats = parts[5]
            head = int(parts[6]) if parts[6] and parts[6] != "_" else 0
            deprel = parts[7] if parts[7] and parts[7] != "_" else "dep"

            feats = {}
            if raw_feats != "_":
                for feat in raw_feats.split("|"):
                    k, v = feat.split("=", 1)
                    feats[k] = v

            current.append((form, upos, feats, head, deprel))

    if current:
        sentences.append(current)

    return sentences


# --- Label indexing ---

def build_label_indices():
    """Build label -> index mappings from labels.py."""
    feat_to_l2i = {}
    for feat, labels in pos_labels.items():
        feat_to_l2i[feat] = {label: i for i, label in enumerate(labels)}
    deprel_l2i = {label: i for i, label in enumerate(dp_labels)}
    return feat_to_l2i, deprel_l2i


# --- Dataset ---

class ConlluDataset(Dataset):
    """Dataset of tokenized CoNLL-U sentences for training."""

    def __init__(self, sentences, tokenizer, feat_to_l2i, deprel_l2i,
                 max_length=512):
        self.items = []
        self.feat_to_l2i = feat_to_l2i
        self.deprel_l2i = deprel_l2i

        for sent in sentences:
            forms = [w[0] for w in sent]
            text = " ".join(forms)
            normalized = strip_accents_and_lowercase(text)

            enc = tokenizer(
                normalized,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc.input_ids.squeeze(0)

            # Build subword-to-word mapping and word mask
            tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
            special_ids = set(tokenizer.all_special_ids)

            word_mask = []
            word_idx = 0
            subword2word = {0: 0}  # CLS -> root

            for j, (tok, tid) in enumerate(zip(tokens, input_ids.tolist())):
                if tid in special_ids:
                    word_mask.append(False)
                    subword2word[j] = 0
                elif tok.startswith("##"):
                    word_mask.append(False)
                    subword2word[j] = word_idx
                else:
                    word_idx += 1
                    word_mask.append(True)
                    subword2word[j] = word_idx

            n_words = sum(word_mask)
            if n_words != len(sent):
                # Tokenizer split doesn't match - skip
                continue

            # Build targets
            pos_targets = {}
            for feat in feat_to_l2i:
                targets = []
                for w in sent:
                    _, upos, feats, _, _ = w
                    if feat == "upos":
                        val = upos
                    else:
                        val = feats.get(feat, "_")
                    idx = feat_to_l2i[feat].get(val, feat_to_l2i[feat].get("_", 0))
                    targets.append(idx)
                pos_targets[feat] = targets

            head_targets = []
            deprel_targets = []
            for w in sent:
                head_targets.append(w[3])  # head index (0 = root)
                deprel_targets.append(
                    self.deprel_l2i.get(w[4], self.deprel_l2i.get("dep", 0))
                )

            self.items.append({
                "input_ids": input_ids,
                "word_mask": word_mask,
                "subword2word": subword2word,
                "pos_targets": pos_targets,
                "head_targets": head_targets,
                "deprel_targets": deprel_targets,
                "n_words": n_words,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    """Collate variable-length sentences into a padded batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    all_word_masks = []
    all_s2w = []
    all_pos_targets = defaultdict(list)
    all_head_targets = []
    all_deprel_targets = []
    all_n_words = []

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = 1
        # Pad word mask to max_len
        padded_mask = item["word_mask"] + [False] * (max_len - len(item["word_mask"]))
        all_word_masks.append(padded_mask)
        all_s2w.append(item["subword2word"])
        for feat, targets in item["pos_targets"].items():
            all_pos_targets[feat].append(targets)
        all_head_targets.append(item["head_targets"])
        all_deprel_targets.append(item["deprel_targets"])
        all_n_words.append(item["n_words"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_masks": all_word_masks,
        "subword2word": all_s2w,
        "pos_targets": dict(all_pos_targets),
        "head_targets": all_head_targets,
        "deprel_targets": all_deprel_targets,
        "n_words": all_n_words,
    }


# --- Training ---

def train_epoch(model, dataloader, optimizer, device, feat_to_l2i,
                scheduler=None, max_grad_norm=1.0):
    """Train one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    ce = nn.CrossEntropyLoss(ignore_index=-1)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pos_logits, arc_scores, rel_scores = model(input_ids, attention_mask)

        loss = torch.tensor(0.0, device=device)
        bs = input_ids.shape[0]

        for b in range(bs):
            mask = batch["word_masks"][b]
            word_positions = [j for j, m in enumerate(mask) if m]
            n_words = batch["n_words"][b]
            if n_words == 0:
                continue

            # POS losses
            for feat in feat_to_l2i:
                logits = pos_logits[feat][b, word_positions]  # (n_words, n_labels)
                targets = torch.tensor(
                    batch["pos_targets"][feat][b], device=device
                )
                loss = loss + ce(logits, targets)

            # Arc loss
            arc = arc_scores[b, word_positions][:, word_positions]  # (n_words, n_words)
            # Include CLS (index 0) as root target
            arc_with_root = arc_scores[b, word_positions]  # (n_words, seq_len)
            head_targets = batch["head_targets"][b]
            # Map word-level head indices to subword positions
            # Head 0 = root = position 0 (CLS)
            s2w = batch["subword2word"][b]
            w2s = {}  # word_idx -> first subword position
            for sw_pos, w_idx in s2w.items():
                if w_idx not in w2s:
                    w2s[w_idx] = sw_pos
            head_subword_targets = []
            for h in head_targets:
                head_subword_targets.append(w2s.get(h, 0))
            head_tensor = torch.tensor(head_subword_targets, device=device)
            loss = loss + ce(arc_with_root, head_tensor)

            # Deprel loss (at predicted head positions)
            deprel_targets = torch.tensor(
                batch["deprel_targets"][b], device=device
            )
            # Gather rel scores at gold head positions
            seq_len = rel_scores.shape[1]
            for w_i, (wp, ht) in enumerate(zip(word_positions, head_subword_targets)):
                rel_logits = rel_scores[b, wp, ht]  # (numrels,)
                loss = loss + ce(
                    rel_logits.unsqueeze(0), deprel_targets[w_i].unsqueeze(0)
                )

        loss = loss / bs
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, feat_to_l2i, deprel_l2i):
    """Evaluate on dev set. Returns accuracy dict."""
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pos_logits, arc_scores, rel_scores = model(input_ids, attention_mask)
        bs = input_ids.shape[0]

        for b in range(bs):
            mask = batch["word_masks"][b]
            word_positions = [j for j, m in enumerate(mask) if m]
            n_words = batch["n_words"][b]

            # POS accuracy
            for feat in feat_to_l2i:
                preds = torch.argmax(pos_logits[feat][b, word_positions], dim=-1)
                targets = torch.tensor(
                    batch["pos_targets"][feat][b], device=device
                )
                correct[feat] += (preds == targets).sum().item()
                total[feat] += n_words

            # Head accuracy
            s2w = batch["subword2word"][b]
            head_preds = torch.argmax(arc_scores[b, word_positions], dim=-1)
            head_word_preds = [s2w.get(hp.item(), 0) for hp in head_preds]
            head_targets = batch["head_targets"][b]
            for hp, ht in zip(head_word_preds, head_targets):
                correct["head"] += int(hp == ht)
                total["head"] += 1

            # Deprel accuracy
            w2s = {}
            for sw_pos, w_idx in s2w.items():
                if w_idx not in w2s:
                    w2s[w_idx] = sw_pos
            deprel_targets = batch["deprel_targets"][b]
            for w_i, (wp, ht) in enumerate(zip(word_positions, head_targets)):
                ht_sw = w2s.get(ht, 0)
                rel_pred = torch.argmax(rel_scores[b, wp, ht_sw]).item()
                correct["deprel"] += int(rel_pred == deprel_targets[w_i])
                total["deprel"] += 1

    return {k: correct[k] / max(total[k], 1) for k in total}


def main():
    parser = argparse.ArgumentParser(description="Train Opla POS+DP heads")
    parser.add_argument("--lang", default="grc", choices=["grc", "el", "med"],
                        help="Language to train (default: grc)")
    parser.add_argument("--data", nargs="+",
                        help="CoNLL-U train files (auto-detected if not specified)")
    parser.add_argument("--dev", nargs="+",
                        help="CoNLL-U dev files (auto-detected if not specified)")
    parser.add_argument("--bert", default=None,
                        help="BERT model name (auto-detected by lang)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output", default=None,
                        help="Output directory for trained weights")
    parser.add_argument("--freeze-bert", action="store_true",
                        help="Freeze BERT weights, train only task heads")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint (.pt file)")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Device: {device}")

    # Auto-detect BERT model
    if args.bert is None:
        if args.lang in ("grc", "med"):
            args.bert = "pranaydeeps/Ancient-Greek-BERT"
        else:
            args.bert = "nlpaueb/bert-base-greek-uncased-v1"
    print(f"BERT: {args.bert}")

    # Auto-detect data files
    data_dir = Path(__file__).parent / "data"
    if args.data is None:
        if args.lang == "grc":
            args.data = sorted(
                data_dir.glob("UD_Ancient_Greek-*/*-train.conllu"))
            # Also include Gorman trees if available
            args.data += sorted(data_dir.glob("Gorman/*-train.conllu"))
        elif args.lang == "med":
            args.data = sorted(data_dir.glob("DiGreC/*-train.conllu"))
        else:
            args.data = sorted(data_dir.glob("UD_Greek-*/*-train.conllu"))
    if args.dev is None:
        if args.lang == "grc":
            args.dev = sorted(
                data_dir.glob("UD_Ancient_Greek-*/*-dev.conllu"))
            args.dev += sorted(data_dir.glob("Gorman/*-dev.conllu"))
        elif args.lang == "med":
            args.dev = sorted(data_dir.glob("DiGreC/*-dev.conllu"))
        else:
            args.dev = sorted(data_dir.glob("UD_Greek-*/*-dev.conllu"))

    if not args.data:
        print("No training data found. Specify --data or place UD treebanks in data/")
        sys.exit(1)

    # Output directory
    if args.output is None:
        args.output = Path(__file__).parent / "weights" / args.lang
    args.output = Path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading training data...")
    train_sents = []
    for f in args.data:
        sents = parse_conllu(f)
        print(f"  {f}: {len(sents)} sentences")
        train_sents.extend(sents)
    print(f"  Total: {len(train_sents)} training sentences")

    dev_sents = []
    if args.dev:
        print(f"Loading dev data...")
        for f in args.dev:
            sents = parse_conllu(f)
            print(f"  {f}: {len(sents)} sentences")
            dev_sents.extend(sents)
        print(f"  Total: {len(dev_sents)} dev sentences")

    # Build label indices
    feat_to_l2i, deprel_l2i = build_label_indices()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert)

    # Build datasets
    print("Building datasets...")
    train_dataset = ConlluDataset(
        train_sents, tokenizer, feat_to_l2i, deprel_l2i
    )
    print(f"  Train: {len(train_dataset)} items (skipped {len(train_sents) - len(train_dataset)} mismatches)")

    dev_dataset = None
    if dev_sents:
        dev_dataset = ConlluDataset(
            dev_sents, tokenizer, feat_to_l2i, deprel_l2i
        )
        print(f"  Dev: {len(dev_dataset)} items")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    dev_loader = None
    if dev_dataset:
        dev_loader = DataLoader(
            dev_dataset, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn
        )

    # Build model (single BERT for joint POS+DP training)
    print(f"\nLoading BERT: {args.bert}")
    bert = AutoModel.from_pretrained(args.bert)
    feat_sizes = {k: len(v) for k, v in pos_labels.items()}
    model = OplaModel(bert, feat_sizes=feat_sizes, num_deprels=len(dp_labels))
    model.to(device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    if args.freeze_bert:
        for param in model.pos_bert.parameters():
            param.requires_grad = False
        print("BERT weights frozen - training task heads only")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = min(500, num_training_steps // 10)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    print(f"LR schedule: {num_warmup_steps} warmup, {num_training_steps} total steps")

    # Train
    total_epochs = start_epoch + args.epochs
    print(f"\nTraining for {args.epochs} epochs (epoch {start_epoch + 1} to {total_epochs})...")
    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device, feat_to_l2i,
                                 scheduler=scheduler)
        elapsed = time.perf_counter() - t0

        msg = f"Epoch {epoch}/{total_epochs}: loss={train_loss:.4f} ({elapsed:.0f}s)"

        if dev_loader:
            acc = evaluate(model, dev_loader, device, feat_to_l2i, deprel_l2i)
            msg += f"  upos={acc['upos']:.3f} head={acc['head']:.3f} deprel={acc['deprel']:.3f}"

        print(msg)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "lang": args.lang,
            "bert_model": args.bert,
            "feat_sizes": feat_sizes,
            "num_deprels": len(dp_labels),
        }
        torch.save(checkpoint, args.output / f"opla_{args.lang}_epoch{epoch}.pt")

    # Save final weights
    final_path = args.output / f"opla_{args.lang}.pt"
    torch.save(checkpoint, final_path)
    print(f"\nSaved final model to {final_path}")

    # Clean up per-epoch checkpoints (keep only final)
    for epoch_file in args.output.glob(f"opla_{args.lang}_epoch*.pt"):
        epoch_file.unlink()
        print(f"  Removed training checkpoint: {epoch_file.name}")

    # Final eval
    if dev_loader:
        print("\nFinal evaluation:")
        acc = evaluate(model, dev_loader, device, feat_to_l2i, deprel_l2i)
        for k, v in sorted(acc.items()):
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
