#!/usr/bin/env python3
"""Convert Gorman AGDT XML treebanks to CoNLL-U format.

Gorman's Greek Dependency Trees use the AGDT (Ancient Greek Dependency
Treebank) annotation scheme. This script converts to UD (Universal
Dependencies) format for training Opla.

AGDT postag: 9-position string (POS, person, number, tense, mood, voice,
gender, case, degree). Relations: SBJ, OBJ, ATR, ADV, PRED, etc.

Usage:
    python convert_gorman.py                          # convert all
    python convert_gorman.py --output data/Gorman/    # custom output
    python convert_gorman.py --stats                  # stats only
"""

import argparse
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

GORMAN_DIR = Path(__file__).parent / "data" / "Greek-Dependency-Trees" / "xml versions"
OUTPUT_DIR = Path(__file__).parent / "data" / "Gorman"

# AGDT POS (position 0 of postag) -> UD UPOS
POS_MAP = {
    "n": "NOUN",
    "v": "VERB",
    "a": "ADJ",
    "d": "ADV",
    "l": "DET",      # article
    "p": "PRON",
    "c": "CCONJ",    # conjunction (refined below)
    "r": "ADP",      # preposition
    "i": "INTJ",
    "m": "NUM",
    "g": "PART",     # particle
    "u": "PUNCT",
    "x": "X",
}

# AGDT relation -> UD deprel
REL_MAP = {
    # Core arguments
    "SBJ": "nsubj",
    "OBJ": "obj",
    "PRED": "root",
    "PNOM": "nsubj",     # predicate nominal -> treat as subject complement

    # Modifiers
    "ATR": "nmod",       # attribute (refined by POS below)
    "ADV": "advmod",     # adverbial (refined by POS below)
    "AuxP": "case",      # preposition
    "AuxC": "mark",      # subordinating conjunction
    "AuxY": "discourse", # sentence adverbial / particle
    "AuxZ": "advmod",    # emphasizing particle
    "APOS": "appos",     # apposition

    # Coordination
    "COORD": "cc",       # coordinator
    "PRED_CO": "conj",
    "SBJ_CO": "conj",
    "OBJ_CO": "conj",
    "ADV_CO": "conj",
    "ATR_CO": "conj",
    "PNOM_CO": "conj",
    "OCOMP_CO": "conj",

    # Punctuation and technical
    "AuxX": "punct",     # comma
    "AuxK": "punct",     # sentence-final punctuation
    "AuxG": "punct",     # other punctuation

    # Rare/complex
    "ExD": "vocative",   # extra-clausal (often vocatives)
    "OCOMP": "xcomp",    # object complement
    "AuxV": "cop",       # auxiliary verb -> copula
}

# AGDT morphology positions
_PERSON = {"1": "1", "2": "2", "3": "3"}
_NUMBER = {"s": "Sing", "p": "Plur", "d": "Dual"}
_TENSE = {
    "p": "Pres", "i": "Past", "a": "Past",
    "f": "Fut", "r": "Past", "l": "Pqp",
    "t": "Fut",
}
_ASPECT = {
    "p": "Imp", "i": "Imp", "a": "Perf",
    "r": "Perf", "l": "Perf", "t": "Perf",
}
_MOOD = {
    "i": "Ind", "s": "Sub", "o": "Opt",
    "m": "Imp", "n": "Inf", "p": "Part",
}
_VOICE = {"a": "Act", "m": "Mid", "p": "Pass", "e": "Mid"}
_GENDER = {"m": "Masc", "f": "Fem", "n": "Neut"}
_CASE = {
    "n": "Nom", "g": "Gen", "d": "Dat",
    "a": "Acc", "v": "Voc", "l": "Loc",
}
_DEGREE = {"c": "Cmp", "s": "Sup"}


def parse_postag(postag: str) -> tuple[str, dict]:
    """Convert AGDT 9-position postag to (UPOS, feats_dict)."""
    if not postag or len(postag) < 1:
        return "X", {}

    pos_char = postag[0]
    upos = POS_MAP.get(pos_char, "X")

    feats = {}
    if len(postag) >= 9:
        if postag[1] in _PERSON:
            feats["Person"] = _PERSON[postag[1]]
        if postag[2] in _NUMBER:
            feats["Number"] = _NUMBER[postag[2]]
        if postag[3] in _TENSE:
            feats["Tense"] = _TENSE[postag[3]]
        if postag[3] in _ASPECT:
            feats["Aspect"] = _ASPECT[postag[3]]
        if postag[4] in _MOOD:
            feats["Mood"] = _MOOD[postag[4]]
            if postag[4] == "p":
                feats["VerbForm"] = "Part"
            elif postag[4] == "n":
                feats["VerbForm"] = "Inf"
            elif postag[4] in ("i", "s", "o", "m"):
                feats["VerbForm"] = "Fin"
        if postag[5] in _VOICE:
            feats["Voice"] = _VOICE[postag[5]]
        if postag[6] in _GENDER:
            feats["Gender"] = _GENDER[postag[6]]
        if postag[7] in _CASE:
            feats["Case"] = _CASE[postag[7]]
        if postag[8] in _DEGREE:
            feats["Degree"] = _DEGREE[postag[8]]

    return upos, feats


def convert_relation(rel: str, upos: str, head_upos: str = "") -> str:
    """Convert AGDT relation to UD deprel, refined by POS."""
    base = rel.rstrip("_CO")

    # Coordinated elements
    if rel.endswith("_CO") and rel != "COORD":
        return "conj"

    # ATR refinement by POS
    if base == "ATR":
        if upos == "ADJ":
            return "amod"
        if upos == "DET":
            return "det"
        if upos == "NUM":
            return "nummod"
        if upos == "VERB":
            return "acl"
        return "nmod"

    # ADV refinement
    if base == "ADV":
        if upos == "VERB":
            return "advcl"
        if upos == "NOUN" or upos == "PROPN":
            return "obl"
        if upos == "ADP":
            return "case"
        return "advmod"

    # SBJ refinement
    if base == "SBJ":
        if upos == "VERB":
            return "csubj"
        return "nsubj"

    # OBJ refinement
    if base == "OBJ":
        if upos == "VERB":
            return "ccomp"
        return "obj"

    # Conjunction refinement: subordinating vs coordinating
    if base == "AuxC":
        return "mark"

    return REL_MAP.get(rel, "dep")


def convert_file(xml_path: Path) -> list[list[dict]]:
    """Convert one Gorman XML file to list of CoNLL-U sentences."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    sentences = []
    for sent_elem in tree.findall(".//sentence"):
        words = []
        for w in sent_elem:
            if w.tag != "word":
                continue

            form = w.get("form", "")
            lemma = w.get("lemma", "")
            postag = w.get("postag", "")
            head = w.get("head", "0")
            relation = w.get("relation", "dep")
            tok_id = w.get("id", "0")

            if not form:
                continue

            # Clean lemma
            if lemma.startswith("punc"):
                lemma = form

            upos, feats = parse_postag(postag)

            # Convert relation
            deprel = convert_relation(relation, upos)

            # Handle root: head=0 means root of sentence
            if head == "0" and upos != "PUNCT":
                deprel = "root"

            # Format features
            if feats:
                feat_str = "|".join(f"{k}={v}" for k, v in sorted(feats.items()))
            else:
                feat_str = "_"

            words.append({
                "id": tok_id,
                "form": form,
                "lemma": lemma,
                "upos": upos,
                "xpos": postag,
                "feats": feat_str,
                "head": head,
                "deprel": deprel,
            })

        if words:
            sentences.append(words)

    return sentences


def write_conllu(sentences: list[list[dict]], output_path: Path,
                 source_file: str = ""):
    """Write sentences in CoNLL-U format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sent in enumerate(sentences, 1):
            f.write(f"# sent_id = {source_file}-{i}\n")
            text = " ".join(w["form"] for w in sent)
            f.write(f"# text = {text}\n")
            for w in sent:
                line = "\t".join([
                    w["id"], w["form"], w["lemma"], w["upos"],
                    w["xpos"], w["feats"], w["head"], w["deprel"],
                    "_", "_",
                ])
                f.write(line + "\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gorman AGDT XML to CoNLL-U")
    parser.add_argument("--input", type=Path, default=GORMAN_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--stats", action="store_true",
                        help="Print stats only, don't write output")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Gorman trees not found at {args.input}")
        return

    xml_files = sorted(args.input.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files in {args.input}")

    all_sentences = []
    total_tokens = 0
    rel_counts = Counter()
    pos_counts = Counter()

    for xml_file in xml_files:
        sentences = convert_file(xml_file)
        for sent in sentences:
            for w in sent:
                total_tokens += 1
                rel_counts[w["deprel"]] += 1
                pos_counts[w["upos"]] += 1
        all_sentences.extend(sentences)

    print(f"Total: {len(all_sentences)} sentences, {total_tokens:,} tokens")

    if args.stats:
        print("\nUPOS distribution:")
        for pos, n in pos_counts.most_common():
            print(f"  {pos:8s} {n:7,}")
        print("\nDeprel distribution (top 15):")
        for rel, n in rel_counts.most_common(15):
            print(f"  {rel:12s} {n:7,}")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    # Split 80/10/10 for train/dev/test
    import random
    random.seed(42)
    indices = list(range(len(all_sentences)))
    random.shuffle(indices)

    n = len(indices)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)

    splits = {
        "train": [all_sentences[i] for i in indices[:train_end]],
        "dev": [all_sentences[i] for i in indices[train_end:dev_end]],
        "test": [all_sentences[i] for i in indices[dev_end:]],
    }

    for split, sents in splits.items():
        out_path = args.output / f"gorman-{split}.conllu"
        tokens = sum(len(s) for s in sents)
        write_conllu(sents, out_path, source_file="gorman")
        print(f"  {split}: {len(sents)} sentences, {tokens:,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
