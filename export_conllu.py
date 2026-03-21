#!/usr/bin/env python3
"""Export Opla's polylas_tags.json to CoNLL-U for gold treebank correction.

Reads Opla's POS/DP output from iliad-align and aligns it with the
original accented Polylas-Riordan source text to produce a CoNLL-U file.

The output is meant for manual correction of dependency heads - fix the
HEAD column, then use the corrected file to fine-tune Opla on Polylas text.

Punctuation tokens are reconstructed from the source text and inserted
with UPOS=PUNCT. Head indices are remapped to account for the inserted
tokens. PUNCT heads default to the sentence root (correct them manually
if needed).

Usage:
    python export_conllu.py                  # all 24 books
    python export_conllu.py 1                # book 1 only
    python export_conllu.py 1 6              # books 1-6
    python export_conllu.py --sample 100     # random 100 sentences
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

# Paths to iliad-align and iliad-murray-riordan (sibling repos)
SCRIPT_DIR = Path(__file__).resolve().parent
ALIGN_DIR = SCRIPT_DIR.parent / "iliad-align"
MR_DIR = SCRIPT_DIR.parent / "iliad-murray-riordan"

TAGS_PATH = ALIGN_DIR / "output" / "polylas_tags.json"
POLYLAS_DIR = MR_DIR / "texts" / "polylas_riordan"
OUTPUT_PATH = SCRIPT_DIR / "data" / "polylas_gold.conllu"


def strip_accents_and_lowercase(s: str) -> str:
    """Match GreekBERT/Opla preprocessing exactly."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).lower()


def align_tags_to_source(text: str, tag_tokens: list[dict]) -> list[dict]:
    """Align Opla tag tokens to the original accented source text.

    Scans the source text character by character, matching each tag token's
    normalized form to recover the original accented span. Characters between
    matched spans (punctuation, spaces) become PUNCT tokens.

    Returns list of {form, upos, feats, head, deprel, is_punct} dicts
    with original accented forms and all inter-token punctuation.
    """
    norm = strip_accents_and_lowercase(text)
    result = []
    pos = 0  # current position in source/norm
    tag_i = 0

    while pos < len(text):
        # Skip whitespace
        if text[pos].isspace():
            pos += 1
            continue

        # Try to match the next tag token at this position
        matched = False
        if tag_i < len(tag_tokens):
            tag_form = tag_tokens[tag_i]["form"]
            tag_len = len(tag_form)
            if norm[pos:pos + tag_len] == tag_form:
                # Found the tag token - extract original accented form
                # Count original chars that correspond to tag_len normalized chars
                orig_start = pos
                norm_consumed = 0
                scan = pos
                while norm_consumed < tag_len and scan < len(text):
                    # Combining marks are stripped in normalization but present
                    # in original. Advance past them without counting.
                    nfd_char = unicodedata.normalize("NFD", text[scan])
                    for c in nfd_char:
                        if unicodedata.category(c) == "Mn":
                            continue
                        norm_consumed += 1
                    scan += 1
                orig_form = text[orig_start:scan]
                tag = tag_tokens[tag_i]
                result.append({
                    "form": orig_form,
                    "upos": tag["upos"],
                    "feats": tag.get("feats", {}),
                    "head": tag["head"],
                    "deprel": tag["deprel"],
                    "lemma": tag.get("lemma", "_"),
                    "is_punct": False,
                    "tag_index": tag_i,
                })
                tag_i += 1
                pos = scan
                matched = True

        if not matched:
            # This character is punctuation (not matched by any tag token)
            # Collect consecutive punctuation chars (but not across spaces)
            punct_start = pos
            while pos < len(text) and not text[pos].isspace():
                # Check if next tag token starts here
                if tag_i < len(tag_tokens):
                    tag_form = tag_tokens[tag_i]["form"]
                    if norm[pos:pos + len(tag_form)] == tag_form:
                        break
                pos += 1
            result.append({
                "form": text[punct_start:pos],
                "is_punct": True,
            })

    return result


def feats_to_str(feats: dict) -> str:
    """Convert feature dict to UD pipe-separated string."""
    if not feats:
        return "_"
    return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))


def build_conllu_sentence(
    book: int, line_num: int, text: str, tag_entry: dict
) -> str:
    """Build a CoNLL-U sentence block from source text and Opla tags.

    Aligns tag tokens to the original accented text via character scanning,
    inserts PUNCT tokens for unmatched punctuation, and remaps HEAD indices.
    """
    tag_tokens = tag_entry["tokens"]
    merged = align_tags_to_source(text, tag_tokens)

    n_matched = sum(1 for t in merged if not t["is_punct"])
    if n_matched != len(tag_tokens):
        print(
            f"  WARN book {book} line {line_num}: matched {n_matched}/"
            f"{len(tag_tokens)} tag tokens",
            file=sys.stderr,
        )

    # Build tag-index -> new-index mapping for head remapping
    # Tag indices are 1-based (0 = root) in polylas_tags.json
    # New indices are 1-based within merged (with PUNCT inserted)
    old_to_new = {0: 0}  # root stays root
    for new_i, tok in enumerate(merged, start=1):
        if not tok["is_punct"] and "tag_index" in tok:
            old_to_new[tok["tag_index"] + 1] = new_i  # tag_index is 0-based

    # Find root token index (for PUNCT head default)
    root_new = 0
    for new_i, tok in enumerate(merged, start=1):
        if not tok["is_punct"] and tok.get("deprel") == "root":
            root_new = new_i
            break

    # Format CoNLL-U lines
    lines = [
        f"# sent_id = book{book:02d}-line{line_num}",
        f"# text = {text}",
    ]
    for new_i, tok in enumerate(merged, start=1):
        form = tok["form"]
        if tok["is_punct"]:
            lines.append(
                f"{new_i}\t{form}\t{form}\tPUNCT\t_\t_\t"
                f"{root_new}\tpunct\t_\t_"
            )
        else:
            upos = tok["upos"]
            feats = feats_to_str(tok.get("feats", {}))
            old_head = tok["head"]
            head = old_to_new.get(old_head, 0)
            deprel = tok["deprel"]
            lemma = tok.get("lemma", "_")
            lines.append(
                f"{new_i}\t{form}\t{lemma}\t{upos}\t_\t{feats}\t"
                f"{head}\t{deprel}\t_\t_"
            )

    lines.append("")  # blank line between sentences
    return "\n".join(lines)


def load_source_lines(book_num: int) -> dict[int, str]:
    """Load original accented lines from Polylas-Riordan source."""
    path = POLYLAS_DIR / f"book_{book_num:02d}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lines = {}
    for seg in data["segments"]:
        # Each segment is one line: start == end
        lines[seg["start"]] = seg["text"]
    return lines


def export_books(books: list[int], sample: int = 0) -> str:
    """Export specified books to CoNLL-U format."""
    with open(TAGS_PATH, encoding="utf-8") as f:
        all_tags = json.load(f)

    sentences = []
    for book_num in books:
        key = str(book_num)
        if key not in all_tags:
            print(f"  Book {book_num}: no tags found, skipping",
                  file=sys.stderr)
            continue

        source_lines = load_source_lines(book_num)
        if not source_lines:
            print(f"  Book {book_num}: no source text found, skipping",
                  file=sys.stderr)
            continue

        for entry in all_tags[key]:
            line_num = entry["line"]
            text = source_lines.get(line_num)
            if text is None:
                continue
            sentences.append((book_num, line_num, text, entry))

    if sample > 0 and sample < len(sentences):
        import random
        random.seed(42)
        sentences = random.sample(sentences, sample)
        sentences.sort(key=lambda x: (x[0], x[1]))

    blocks = []
    for book_num, line_num, text, entry in sentences:
        blocks.append(build_conllu_sentence(book_num, line_num, text, entry))

    return "\n".join(blocks)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Export Opla tags to CoNLL-U for gold treebank correction"
    )
    parser.add_argument("start", nargs="?", type=int, default=None,
                        help="First book (default: 1)")
    parser.add_argument("end", nargs="?", type=int, default=None,
                        help="Last book (default: same as start, or 24)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Random sample of N sentences (0 = all)")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    if args.start is None:
        args.start = 1
        args.end = 24  # no positional args = all books
    elif args.end is None:
        args.end = args.start  # single book

    books = list(range(args.start, args.end + 1))
    out_path = Path(args.output) if args.output else OUTPUT_PATH

    print(f"Exporting books {args.start}-{args.end}"
          + (f" (sample {args.sample})" if args.sample else ""))

    conllu = export_books(books, sample=args.sample)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(conllu, encoding="utf-8")

    n_sents = conllu.count("# sent_id")
    print(f"Wrote {n_sents} sentences to {out_path}")


if __name__ == "__main__":
    main()
