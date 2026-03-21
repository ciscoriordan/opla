#!/usr/bin/env python3
"""Merge enjambed verse lines into proper multi-line CoNLL-U sentences.

When a verse line's main verb depends on a word in a different line
(relative clauses, adverbial clauses, etc.), the two lines must be
one CoNLL-U sentence for the head pointer to work.

Reads polylas_tags.json and the Polylas-Riordan source text to build
merged sentences, then re-exports them with correct head indices.

Usage:
    python merge_enjambed.py data/polylas_gold_deep.conllu
"""

import json
import sys
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ALIGN_DIR = SCRIPT_DIR.parent / "iliad-align"
MR_DIR = SCRIPT_DIR.parent / "iliad-murray-riordan"

TAGS_PATH = ALIGN_DIR / "output" / "polylas_tags.json"
POLYLAS_DIR = MR_DIR / "texts" / "polylas_riordan"


def strip_accents_and_lowercase(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).lower()


def parse_conllu(path):
    sents = {}
    current_meta = []
    current_tokens = []
    current_id = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id"):
                current_id = line.split("=", 1)[1].strip()
                current_meta = [line]
            elif line.startswith("#"):
                current_meta.append(line)
            elif not line:
                if current_id and current_tokens:
                    sents[current_id] = {"meta": current_meta, "tokens": current_tokens}
                current_tokens = []
                current_id = None
                current_meta = []
            else:
                current_tokens.append(line.split("\t"))
        if current_id and current_tokens:
            sents[current_id] = {"meta": current_meta, "tokens": current_tokens}
    return sents


def write_conllu(sents_list, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in sents_list:
            for m in s["meta"]:
                f.write(m + "\n")
            for t in s["tokens"]:
                f.write("\t".join(t) + "\n")
            f.write("\n")


def load_source_line(book, line):
    path = POLYLAS_DIR / f"book_{book:02d}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for seg in data["segments"]:
        if seg["start"] == line:
            return seg["text"]
    return None


def load_tags_line(all_tags, book, line):
    key = str(book)
    if key not in all_tags:
        return None
    for entry in all_tags[key]:
        if entry["line"] == line:
            return entry
    return None


# Lines that need merging: (orphan_sid, parent_line, deprel_of_orphan_root)
MERGES = [
    ("book03-line167", 166, "acl:relcl"),
    ("book06-line126", 125, "advcl"),
    ("book10-line239", 238, "advcl"),
    ("book14-line510", 509, "advcl"),
    ("book18-line124", 122, "csubj"),
    ("book18-line176", 175, "csubj"),
    ("book21-line346", 345, "advcl"),
    ("book22-line418", 417, "parataxis"),
    ("book24-line231", 230, "advmod"),
]


def build_merged_sentence(orphan_sent, parent_tokens, parent_text,
                          parent_book, parent_line, orphan_deprel):
    """Merge parent line tokens + orphan line tokens into one sentence.

    Renumbers all token IDs and remaps heads.
    The orphan line's root verb gets attached to the parent's root/last-verb
    with the specified deprel.
    """
    orphan_tokens = orphan_sent["tokens"]
    n_parent = len(parent_tokens)
    n_orphan = len(orphan_tokens)

    # Find parent root (the token the orphan should attach to)
    parent_root_id = 0
    # Prefer the last verb or noun in parent for attachment
    for t in parent_tokens:
        if t[7] == "root":
            parent_root_id = int(t[0])
            break
    if not parent_root_id:
        for t in reversed(parent_tokens):
            if t[3] == "VERB":
                parent_root_id = int(t[0])
                break
    if not parent_root_id:
        for t in reversed(parent_tokens):
            if t[3] in ("NOUN", "PROPN"):
                parent_root_id = int(t[0])
                break
    if not parent_root_id:
        parent_root_id = 1  # fallback to first token

    # Build merged token list
    merged = []

    # Parent tokens keep their IDs (1-based)
    for t in parent_tokens:
        merged.append(list(t))

    # Orphan tokens get IDs offset by n_parent
    orphan_root_new_id = None
    for t in orphan_tokens:
        new_t = list(t)
        old_id = int(t[0])
        new_id = old_id + n_parent
        new_t[0] = str(new_id)

        # Remap head
        old_head = int(t[6])
        if old_head == 0:
            if t[7] in ("root", "punct"):
                # Orphan root -> attach to parent root with specified deprel
                if t[7] == "root" or (t[7] != "punct" and t[3] == "VERB"):
                    new_t[6] = str(parent_root_id)
                    new_t[7] = orphan_deprel
                    orphan_root_new_id = new_id
                elif t[7] == "punct":
                    # Attach punct to the new merged root
                    new_t[6] = str(parent_root_id)
                else:
                    new_t[6] = str(parent_root_id)
            else:
                # Non-root with head=0: attach to parent root
                new_t[6] = str(parent_root_id)
        else:
            new_t[6] = str(old_head + n_parent)

        merged.append(new_t)

    # Build metadata
    orphan_meta = orphan_sent["meta"]
    orphan_sid = None
    orphan_text = None
    for m in orphan_meta:
        if m.startswith("# sent_id"):
            orphan_sid = m.split("=", 1)[1].strip()
        if m.startswith("# text"):
            orphan_text = m.split("=", 1)[1].strip()

    new_sid = f"book{parent_book:02d}-line{parent_line}+{orphan_sid.split('-')[1]}"
    combined_text = f"{parent_text} | {orphan_text}" if orphan_text else parent_text

    meta = [
        f"# sent_id = {new_sid}",
        f"# text = {combined_text}",
    ]

    return {"meta": meta, "tokens": merged}, orphan_sid


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    sents = parse_conllu(args.input)
    all_tags = json.load(open(TAGS_PATH, encoding="utf-8"))

    # Import export_conllu's alignment function to build parent tokens
    sys.path.insert(0, str(SCRIPT_DIR))
    from export_conllu import align_tags_to_source, feats_to_str

    merged_sids = set()
    new_sents = {}

    for orphan_sid, parent_line_num, orphan_deprel in MERGES:
        if orphan_sid not in sents:
            print(f"  SKIP {orphan_sid}: not in input file")
            continue

        book = int(orphan_sid.split("-")[0].replace("book", ""))

        # Get parent line data
        parent_text = load_source_line(book, parent_line_num)
        parent_tags = load_tags_line(all_tags, book, parent_line_num)
        if not parent_text or not parent_tags:
            print(f"  SKIP {orphan_sid}: parent line {parent_line_num} not found")
            continue

        # Build parent tokens using the same alignment as export_conllu
        parent_aligned = align_tags_to_source(parent_text, parent_tags["tokens"])

        # Convert to CoNLL-U token format
        parent_tokens = []
        tag_idx_to_new = {0: 0}
        new_i = 0
        for tok in parent_aligned:
            new_i += 1
            if tok["is_punct"]:
                parent_tokens.append([
                    str(new_i), tok["form"], tok["form"], "PUNCT", "_", "_",
                    "0", "punct", "_", "_"
                ])
            else:
                tag_idx = tok.get("tag_index", -1)
                if tag_idx >= 0:
                    tag_idx_to_new[tag_idx + 1] = new_i

                feats = feats_to_str(tok.get("feats", {}))
                old_head = tok.get("head", 0)
                deprel = tok.get("deprel", "dep")
                lemma = tok.get("lemma", "_")

                parent_tokens.append([
                    str(new_i), tok["form"], lemma, tok.get("upos", "X"),
                    "_", feats, "0", deprel, "_", "_"
                ])

        # Remap parent heads using tag index mapping
        root_new = 0
        for pt in parent_tokens:
            if pt[3] != "PUNCT":
                # Find original tag index for this token
                pass

        # Actually, let's use a simpler approach: re-use the tag heads
        # and remap through tag_idx_to_new
        content_i = 0
        for pt in parent_tokens:
            if pt[3] == "PUNCT":
                continue
            tag_tok = parent_tags["tokens"][content_i]
            old_head = tag_tok["head"]
            new_head = tag_idx_to_new.get(old_head, 0)
            pt[6] = str(new_head)
            pt[7] = tag_tok["deprel"]
            if tag_tok["deprel"] == "root":
                root_new = int(pt[0])
            content_i += 1

        # Set PUNCT heads to root
        for pt in parent_tokens:
            if pt[3] == "PUNCT" and int(pt[6]) == 0:
                pt[6] = str(root_new) if root_new else "0"

        # Build merged sentence
        merged, _ = build_merged_sentence(
            sents[orphan_sid], parent_tokens, parent_text,
            book, parent_line_num, orphan_deprel
        )
        new_sents[orphan_sid] = merged
        merged_sids.add(orphan_sid)
        print(f"  MERGED {orphan_sid} with line {parent_line_num} ({orphan_deprel})")

    # Rebuild output: replace orphan sentences with merged ones,
    # keep everything else
    output = []
    for sid, s in sents.items():
        if sid in merged_sids:
            output.append(new_sents[sid])
        else:
            output.append(s)

    out_path = args.output or args.input
    write_conllu(output, out_path)
    print(f"\n{len(merged_sids)} sentences merged -> {out_path}")


if __name__ == "__main__":
    main()
