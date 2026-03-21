#!/usr/bin/env python3
"""First-pass heuristic correction of dependency heads in CoNLL-U.

Fixes:
  1. Self-loops: token HEAD pointing to itself
  2. DET attachment: articles should point to nearest NOUN/PROPN
  3. Out-of-range heads

Logs every change for review. Run before manual correction in UD Annotatrix.

Usage:
    python fix_heads.py data/polylas_gold.conllu
    python fix_heads.py data/polylas_gold.conllu -o data/polylas_gold_fixed.conllu
"""

import sys
from pathlib import Path


def parse_conllu(path):
    """Parse CoNLL-U into list of sentences, each a dict with meta + tokens."""
    sents = []
    current = {"meta": [], "tokens": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                current["meta"].append(line)
            elif not line:
                if current["tokens"]:
                    sents.append(current)
                current = {"meta": [], "tokens": []}
            else:
                current["tokens"].append(line.split("\t"))
        if current["tokens"]:
            sents.append(current)
    return sents


def write_conllu(sents, path):
    """Write sentences back to CoNLL-U."""
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            for m in s["meta"]:
                f.write(m + "\n")
            for t in s["tokens"]:
                f.write("\t".join(t) + "\n")
            f.write("\n")


def find_nearest(tokens, pos, target_upos, direction="right", skip_self=True):
    """Find nearest token with target UPOS in given direction.

    Args:
        tokens: list of CoNLL-U token arrays
        pos: 0-based index of current token
        target_upos: set of UPOS tags to search for
        direction: "right", "left", or "both"
        skip_self: skip the token at pos
    Returns:
        1-based token ID, or 0 if not found
    """
    n = len(tokens)

    if direction in ("right", "both"):
        for i in range(pos + 1, n):
            if tokens[i][3] in target_upos:
                return int(tokens[i][0])

    if direction in ("left", "both"):
        for i in range(pos - 1, -1, -1):
            if tokens[i][3] in target_upos:
                return int(tokens[i][0])

    return 0


def find_nearest_either(tokens, pos, target_upos):
    """Find nearest token with target UPOS, preferring right then left."""
    n = len(tokens)
    best = None
    best_dist = n + 1
    for i in range(n):
        if i == pos:
            continue
        if tokens[i][3] in target_upos:
            dist = abs(i - pos)
            # Prefer right (forward) by giving it a small bonus
            if i > pos:
                dist -= 0.1
            if dist < best_dist:
                best_dist = dist
                best = int(tokens[i][0])
    return best or 0


def get_sent_id(sent):
    for m in sent["meta"]:
        if m.startswith("# sent_id"):
            return m.split("=", 1)[1].strip()
    return "?"


def fix_sentence(sent, log):
    """Apply heuristic head fixes to a sentence. Returns number of fixes."""
    tokens = sent["tokens"]
    n = len(tokens)
    sent_id = get_sent_id(sent)
    fixes = 0

    # Find root token for fallback
    root_id = 0
    for t in tokens:
        if t[7] == "root":
            root_id = int(t[0])
            break
    if not root_id:
        # Use first verb as pseudo-root
        for t in tokens:
            if t[3] == "VERB":
                root_id = int(t[0])
                break

    nouns = {"NOUN", "PROPN"}
    noun_adj = {"NOUN", "PROPN", "ADJ", "NUM"}
    verbs = {"VERB", "AUX"}

    for idx, t in enumerate(tokens):
        tid = int(t[0])
        upos = t[3]
        head = int(t[6])
        deprel = t[7]
        form = t[1]
        old_head = head

        # --- Fix 1: Out-of-range heads ---
        if head > n or head < 0:
            head = root_id or 0
            t[6] = str(head)
            log.append(f"  {sent_id} #{tid} '{form}': head {old_head} out of range -> {head}")
            fixes += 1
            continue

        # --- Fix 2: Self-loops ---
        if head == tid:
            if upos == "DET":
                head = find_nearest(tokens, idx, noun_adj, "right")
            elif upos in ("ADJ", "NUM"):
                head = find_nearest_either(tokens, idx, nouns)
            elif upos == "ADV":
                head = find_nearest_either(tokens, idx, verbs)
                if not head:
                    head = find_nearest_either(tokens, idx, nouns | {"ADJ"})
            elif upos == "NOUN":
                # Noun self-loop: likely a genitive modifier or apposition
                head = find_nearest_either(tokens, idx, nouns)
                if not head:
                    head = find_nearest_either(tokens, idx, verbs)
            elif upos == "PRON":
                head = find_nearest_either(tokens, idx, verbs)
                if not head:
                    head = find_nearest_either(tokens, idx, nouns)
            elif upos in ("VERB", "AUX"):
                head = root_id if root_id != tid else 0
            elif upos == "ADP":
                head = find_nearest(tokens, idx, noun_adj, "right")
            elif upos == "SCONJ":
                head = find_nearest(tokens, idx, verbs, "right")
            else:
                head = root_id if root_id != tid else 0

            if head == tid or head == 0:
                head = root_id if root_id != tid else 0
            t[6] = str(head)
            log.append(f"  {sent_id} #{tid} '{form}' ({upos}): self-loop -> {head}")
            fixes += 1
            continue

        # --- Fix 3: DET pointing to wrong head ---
        if upos == "DET" and deprel == "det" and head > 0:
            head_upos = tokens[head - 1][3]
            if head_upos not in noun_adj and head_upos != "PRON":
                new_head = find_nearest(tokens, idx, noun_adj, "right")
                if not new_head:
                    new_head = find_nearest(tokens, idx, noun_adj, "left")
                if new_head and new_head != tid:
                    t[6] = str(new_head)
                    log.append(
                        f"  {sent_id} #{tid} '{form}': det {old_head}"
                        f" ({head_upos}) -> {new_head}"
                        f" ({tokens[new_head - 1][3]})"
                    )
                    fixes += 1

    return fixes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix dependency heads in CoNLL-U")
    parser.add_argument("input", help="Input CoNLL-U file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file (default: overwrite input)")
    args = parser.parse_args()

    sents = parse_conllu(args.input)
    log = []
    total_fixes = 0

    for sent in sents:
        total_fixes += fix_sentence(sent, log)

    out_path = args.output or args.input
    write_conllu(sents, out_path)

    for line in log:
        print(line)
    print(f"\n{total_fixes} fixes in {len(sents)} sentences -> {out_path}")


if __name__ == "__main__":
    main()
