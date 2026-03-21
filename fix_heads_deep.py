#!/usr/bin/env python3
"""Deep linguistic correction of Opla's MG dependency heads.

Goes beyond structural fixes (self-loops, bad DET) to fix semantic errors:
- Genitive modifiers attached to wrong nouns
- Head=0 on non-root tokens (enjambed lines missing root)
- Prepositional phrases attached wrongly
- Coordination errors
- Clitic pronoun placement
- Adjective/adverb attachment

Reads and corrects a CoNLL-U file in place.

Usage:
    python fix_heads_deep.py data/polylas_gold_fixed.conllu
"""

import sys
from pathlib import Path


def parse_conllu(path):
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
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            for m in s["meta"]:
                f.write(m + "\n")
            for t in s["tokens"]:
                f.write("\t".join(t) + "\n")
            f.write("\n")


def get_sent_id(sent):
    for m in sent["meta"]:
        if m.startswith("# sent_id"):
            return m.split("=", 1)[1].strip()
    return "?"


def tid(t):
    return int(t[0])


def form(t):
    return t[1]


def upos(t):
    return t[3]


def head(t):
    return int(t[6])


def deprel(t):
    return t[7]


def set_head(t, h):
    t[6] = str(h)


def set_deprel(t, d):
    t[7] = d


def find_root(tokens):
    for t in tokens:
        if deprel(t) == "root":
            return tid(t)
    return 0


def find_verb(tokens):
    """Find the main verb (first finite verb, or first verb)."""
    for t in tokens:
        if upos(t) == "VERB" and "VerbForm=Fin" in t[5]:
            return tid(t)
    for t in tokens:
        if upos(t) == "VERB":
            return tid(t)
    return 0


def children_of(tokens, h):
    """Get all tokens whose head is h."""
    return [t for t in tokens if head(t) == h]


def fix_sentence(sent, log):
    tokens = sent["tokens"]
    sid = get_sent_id(sent)
    n = len(tokens)
    fixes = 0
    root_id = find_root(tokens)
    verb_id = find_verb(tokens)

    for idx, t in enumerate(tokens):
        i = tid(t)
        h = head(t)
        u = upos(t)
        d = deprel(t)
        f = form(t)

        # --- Fix: head=0 on non-root tokens ---
        # These are enjambed lines where Opla couldn't find a root.
        # Attach orphan tokens to the sentence verb or root.
        if h == 0 and d != "root" and d != "punct":
            target = root_id or verb_id
            if target and target != i:
                # DET with head=0: attach to nearest following noun
                if u == "DET":
                    for j in range(idx + 1, n):
                        if upos(tokens[j]) in ("NOUN", "PROPN", "ADJ"):
                            target = tid(tokens[j])
                            break
                # ADJ/AMOD with head=0: attach to nearest noun
                elif u == "ADJ":
                    # Look right first, then left
                    for j in range(idx + 1, n):
                        if upos(tokens[j]) in ("NOUN", "PROPN"):
                            target = tid(tokens[j])
                            break
                    else:
                        for j in range(idx - 1, -1, -1):
                            if upos(tokens[j]) in ("NOUN", "PROPN"):
                                target = tid(tokens[j])
                                break
                elif u == "ADP":
                    # Preposition: attach to following noun
                    for j in range(idx + 1, n):
                        if upos(tokens[j]) in ("NOUN", "PROPN"):
                            target = tid(tokens[j])
                            d = "case"
                            set_deprel(t, d)
                            break
                elif u == "CCONJ":
                    # Coordinator: find the next content word
                    for j in range(idx + 1, n):
                        if upos(tokens[j]) in ("NOUN", "PROPN", "VERB", "ADJ"):
                            target = tid(tokens[j])
                            d = "cc"
                            set_deprel(t, d)
                            break
                elif u == "PRON":
                    target = root_id or verb_id or 0
                elif u in ("ADV", "PART"):
                    target = root_id or verb_id or 0

                if target and target != i:
                    set_head(t, target)
                    log.append(f"  {sid} #{i} '{f}' ({u}): head 0 -> {target}")
                    fixes += 1

        # --- Fix: DET with deprel=det but head is not a noun ---
        # Already handled by fix_heads.py, but check for remaining cases
        if u == "DET" and d == "det" and h > 0 and h <= n:
            head_u = upos(tokens[h - 1])
            if head_u not in ("NOUN", "PROPN", "ADJ", "NUM", "PRON"):
                # Find nearest noun to the right
                for j in range(idx + 1, n):
                    if upos(tokens[j]) in ("NOUN", "PROPN", "ADJ"):
                        set_head(t, tid(tokens[j]))
                        log.append(
                            f"  {sid} #{i} '{f}': det->{head_u} -> "
                            f"{tid(tokens[j])} ({upos(tokens[j])})"
                        )
                        fixes += 1
                        break

        # --- Fix: "με" (ADP) with deprel=case should point to following noun ---
        if u == "ADP" and d == "case" and h > 0 and h <= n:
            head_u = upos(tokens[h - 1])
            if head_u not in ("NOUN", "PROPN", "ADJ", "PRON", "NUM", "X"):
                for j in range(idx + 1, n):
                    if upos(tokens[j]) in ("NOUN", "PROPN", "PRON"):
                        set_head(t, tid(tokens[j]))
                        log.append(
                            f"  {sid} #{i} '{f}': case->{head_u} -> "
                            f"{tid(tokens[j])}"
                        )
                        fixes += 1
                        break

        # --- Fix: genitive DET (του/της/των) should attach to following noun ---
        if u == "DET" and d == "det" and h > 0 and h <= n:
            feats = t[5]
            if "Case=Gen" in feats:
                head_t = tokens[h - 1]
                head_feats = head_t[5]
                # If head is not genitive but there's a genitive noun nearby, fix
                if "Case=Gen" not in head_feats:
                    for j in range(idx + 1, min(idx + 4, n)):
                        if upos(tokens[j]) in ("NOUN", "PROPN", "ADJ"):
                            if "Case=Gen" in tokens[j][5]:
                                set_head(t, tid(tokens[j]))
                                log.append(
                                    f"  {sid} #{i} '{f}': gen-det -> "
                                    f"{tid(tokens[j])} '{form(tokens[j])}'"
                                )
                                fixes += 1
                                break

        # --- Fix: PUNCT with head=0 should point to root ---
        if u == "PUNCT" and h == 0:
            target = root_id or verb_id
            if target:
                set_head(t, target)
                # Don't log these - too noisy

        # --- Fix: "στο/στη/στον/στην" tagged as DET with case deprel ---
        # These are preposition+article contractions, should be case
        if d == "case" and u == "DET":
            # This is correct behavior for στο/στη etc.
            pass

        # --- Fix: coordinated verbs with head=0 ---
        # "κι ... στέρξαν" where the second verb should attach to first
        if u == "VERB" and h == 0 and d != "root":
            # Look for a preceding CCONJ that might coordinate us
            for j in range(idx - 1, -1, -1):
                if upos(tokens[j]) == "CCONJ":
                    # Find the first verb before the CCONJ
                    for k in range(j - 1, -1, -1):
                        if upos(tokens[k]) == "VERB":
                            set_head(t, tid(tokens[k]))
                            set_deprel(t, "conj")
                            log.append(
                                f"  {sid} #{i} '{f}': verb h=0 -> "
                                f"{tid(tokens[k])} '{form(tokens[k])}' (conj)"
                            )
                            fixes += 1
                            break
                    break

        # --- Fix: "ο/η" (article) tagged as PROPN should be DET ---
        # Not a head fix but a tag fix that helps downstream
        # Skip - not our job here, this is for POS correction

    # --- Second pass: fix orphan CCONJ/SCONJ whose head is still 0 ---
    root_id = find_root(tokens)  # recalc after fixes
    for t in tokens:
        if head(t) == 0 and deprel(t) not in ("root", "punct"):
            target = root_id or verb_id
            if target and target != tid(t):
                set_head(t, target)

    return fixes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    sents = parse_conllu(args.input)
    log = []
    total = 0
    for s in sents:
        total += fix_sentence(s, log)

    out = args.output or args.input
    write_conllu(sents, out)
    for line in log:
        print(line)
    print(f"\n{total} deep fixes in {len(sents)} sentences -> {out}")


if __name__ == "__main__":
    main()
