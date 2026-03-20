#!/usr/bin/env python3
"""Convert DiGreC PROIEL XML treebank to CoNLL-U format for Opla training.

DiGreC uses PROIEL annotation scheme. This script converts:
- PROIEL POS tags -> UD UPOS tags
- PROIEL dependency relations -> UD deprels
- PROIEL morphology string -> UD feature format

The morphology string is 10 chars: person, number, tense, mood, voice,
gender, case, degree, strength, inflection.

Usage:
    python convert_digrec.py                    # convert to data/DiGreC/
    python convert_digrec.py --split 0.8 0.1    # 80% train, 10% dev, 10% test
"""

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# PROIEL POS -> UD UPOS mapping
POS_MAP = {
    "A-": "ADJ",
    "Df": "ADV",
    "Dq": "ADV",
    "Du": "ADV",
    "S-": "DET",
    "Ma": "NUM",
    "Mo": "NUM",
    "Nb": "NOUN",
    "Ne": "PROPN",
    "C-": "CCONJ",  # refined to SCONJ in convert_sentence where applicable
    "Pd": "DET",    # demonstrative pronoun -> DET in UD
    "Px": "PRON",
    "Pi": "PRON",
    "Pp": "PRON",
    "Pk": "PRON",
    "Ps": "DET",    # possessive -> DET in UD
    "Pt": "DET",
    "Pc": "PRON",
    "Pr": "PRON",   # relative pronoun
    "Py": "PRON",   # quantifier
    "R-": "ADP",
    "N-": "PART",
    "I-": "INTJ",
    "F-": "X",
    "G-": "PART",   # subjunction particle
    "V-": "VERB",
}

# PROIEL relation -> UD deprel mapping (base, refined by POS in convert_sentence)
REL_MAP = {
    "pred": "root",
    "sub": "nsubj",
    "obj": "obj",
    "obl": "obl",
    "adv": "advmod",
    "atr": "amod",
    "aux": "dep",     # refined by POS below
    "comp": "ccomp",
    "xadv": "advcl",
    "xobj": "xcomp",
    "apos": "appos",
    "part": "obl",
    "narg": "nmod",
    "ag": "obl:agent",
    "parpred": "parataxis",
    "voc": "vocative",
    "arg": "obj",
    "rel": "acl:relcl",
    "adnom": "nmod",
    "expl": "expl",
    "nonsub": "obl",
    "per": "obl",
    "pid": "nsubj",
    "xsub": "nsubj",
}

# Discourse particles (PROIEL aux+Df that should be UD discourse, not advmod)
_DISCOURSE_LEMMAS = {
    "δέ", "μέν", "γάρ", "δή", "οὖν", "γε", "ἄρα", "περ", "τοίνυν",
    "μέντοι", "ἀτάρ", "αὐτάρ", "μήν", "θήν", "ἤτοι",
}
# Negation particles
_NEGATION_LEMMAS = {"οὐ", "μή", "οὐκ", "οὐχ", "μηδ"}
# Modal particles -> UD aux
_MODAL_LEMMAS = {"ἄν", "κε", "κέν"}

# Morphology string positions (10 chars)
# 0: person (1,2,3,-)
# 1: number (s,d,p,-)
# 2: tense (p,i,r,s,t,l,f,a,-)
# 3: mood (i,s,o,n,p,e,g,d,-)
# 4: voice (a,m,p,e,-)
# 5: gender (m,f,n,-)
# 6: case (n,g,d,a,v,b,c,l,i,-)
# 7: degree (p,c,s,-)
# 8: strength/definiteness (w,s,t,-)
# 9: inflection (i,n,-)

PERSON_MAP = {"1": "1", "2": "2", "3": "3"}
NUMBER_MAP = {"s": "Sing", "d": "Dual", "p": "Plur"}
TENSE_MAP = {"p": "Pres", "i": "Past", "r": "Perf", "s": "Past",
             "t": "Pqp", "l": "Pqp", "f": "Fut", "a": "Past"}
MOOD_MAP = {"i": "Ind", "s": "Sub", "o": "Opt", "n": "Inf",
            "p": "Part", "e": "Imp", "g": "Ger", "d": "Ind"}
VOICE_MAP = {"a": "Act", "m": "Mid", "p": "Pass", "e": "Mid"}
GENDER_MAP = {"m": "Masc", "f": "Fem", "n": "Neut"}
CASE_MAP = {"n": "Nom", "g": "Gen", "d": "Dat", "a": "Acc",
            "v": "Voc", "b": "Abl", "c": "Loc", "l": "Loc", "i": "Ins"}
DEGREE_MAP = {"p": "Pos", "c": "Cmp", "s": "Sup"}


def parse_morphology(morph_str):
    """Convert PROIEL 10-char morphology string to UD feature dict."""
    if not morph_str or len(morph_str) < 10:
        return {}

    feats = {}
    if morph_str[0] in PERSON_MAP:
        feats["Person"] = PERSON_MAP[morph_str[0]]
    if morph_str[1] in NUMBER_MAP:
        feats["Number"] = NUMBER_MAP[morph_str[1]]
    if morph_str[2] in TENSE_MAP:
        feats["Tense"] = TENSE_MAP[morph_str[2]]
    if morph_str[3] in MOOD_MAP:
        val = MOOD_MAP[morph_str[3]]
        if val in ("Part", "Inf", "Ger"):
            feats["VerbForm"] = val
        else:
            feats["VerbForm"] = "Fin"
            feats["Mood"] = val
    if morph_str[4] in VOICE_MAP:
        feats["Voice"] = VOICE_MAP[morph_str[4]]
    if morph_str[5] in GENDER_MAP:
        feats["Gender"] = GENDER_MAP[morph_str[5]]
    if morph_str[6] in CASE_MAP:
        feats["Case"] = CASE_MAP[morph_str[6]]
    if morph_str[7] in DEGREE_MAP:
        feats["Degree"] = DEGREE_MAP[morph_str[7]]

    return feats


def feats_to_str(feats):
    """Convert feature dict to CoNLL-U string."""
    if not feats:
        return "_"
    return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))


def convert_sentence(sent_elem):
    """Convert a PROIEL XML sentence to CoNLL-U lines."""
    tokens = [t for t in sent_elem if t.tag == "token"]
    if not tokens:
        return None

    # Build id mapping (PROIEL uses arbitrary IDs, CoNLL-U uses 1-indexed)
    old_to_new = {}
    for i, t in enumerate(tokens, 1):
        old_to_new[t.get("id")] = i

    lines = []
    parsed = []
    for i, t in enumerate(tokens, 1):
        form = t.get("form", "_")
        lemma = t.get("lemma", "_")
        proiel_pos = t.get("part-of-speech", "?")
        upos = POS_MAP.get(proiel_pos, "X")
        # Refine C- to SCONJ for subordinating conjunctions
        if proiel_pos == "C-" and lemma in (
            "ὅτι", "ὡς", "ἵνα", "ὅπως", "εἰ", "ἐάν", "ὅταν",
            "ἐπεί", "ἕως", "πρίν", "ὥστε", "εἴτε", "εἴπερ",
        ):
            upos = "SCONJ"
        morph = t.get("morphology", "")
        feats = parse_morphology(morph)
        head_id = t.get("head-id", "")
        relation = t.get("relation", "dep")
        deprel = REL_MAP.get(relation, "dep")

        # Map head ID
        if head_id and head_id in old_to_new:
            head = old_to_new[head_id]
        else:
            head = 0  # root or missing head

        # --- Refined deprel mapping by POS and lemma ---

        # PROIEL "aux" covers determiners, particles, conjunctions, etc.
        if relation == "aux":
            if proiel_pos == "S-":
                deprel = "det"
            elif proiel_pos in ("Pd", "Ps", "Pt"):
                deprel = "det"
            elif proiel_pos == "C-":
                deprel = "cc"
            elif proiel_pos == "R-":
                deprel = "case"
            elif proiel_pos == "I-":
                deprel = "discourse"
            elif proiel_pos == "V-":
                # Auxiliary verbs (εἰμί as copula, etc.)
                deprel = "cop" if lemma == "εἰμί" else "aux"
            elif proiel_pos in ("Df", "Dq", "Du"):
                if lemma in _DISCOURSE_LEMMAS:
                    deprel = "discourse"
                elif lemma in _NEGATION_LEMMAS:
                    deprel = "advmod"
                elif lemma in _MODAL_LEMMAS:
                    deprel = "aux"
                elif lemma == "καί":
                    deprel = "cc"
                else:
                    deprel = "advmod"
            elif proiel_pos == "G-":
                deprel = "mark"  # subjunction particles
            else:
                deprel = "dep"

        # PROIEL "adv" - adverbial modifiers
        if relation == "adv":
            if proiel_pos == "R-":
                deprel = "case"
            elif proiel_pos == "G-":
                deprel = "mark"  # subordinators (εἰ, ὡς, ἐπεί, ἵνα...)
            elif proiel_pos == "C-":
                if lemma in ("ὅτι", "ὡς", "ἵνα", "ὅπως", "εἰ", "ἐάν",
                              "ὅταν", "ἐπεί", "ἕως", "πρίν", "ὥστε"):
                    deprel = "mark"
                else:
                    deprel = "cc"
            elif proiel_pos == "V-":
                if morph and len(morph) >= 4 and morph[3] == "p":
                    deprel = "advcl"  # participle as adverbial
                elif morph and len(morph) >= 4 and morph[3] == "n":
                    deprel = "advcl"  # infinitive as adverbial
                else:
                    deprel = "advcl"
            elif proiel_pos in ("Nb", "Ne"):
                deprel = "obl"  # nouns as adverbial -> oblique
            elif proiel_pos in ("A-",):
                deprel = "advmod"
            elif proiel_pos in ("Pr",):
                deprel = "advcl"  # relative pronoun -> clause
            elif proiel_pos in ("Pd", "Px", "Py", "Pi", "Pp"):
                deprel = "obl"  # pronouns as adverbial -> oblique
            else:
                deprel = "advmod"

        # PROIEL "atr" - attributive modifiers
        if relation == "atr":
            if proiel_pos == "A-":
                deprel = "amod"
            elif proiel_pos == "R-":
                deprel = "case"
            elif proiel_pos in ("S-", "Pd", "Ps", "Pt"):
                deprel = "det"
            elif proiel_pos in ("Nb", "Ne"):
                deprel = "nmod"
            elif proiel_pos == "V-":
                if morph and len(morph) >= 4 and morph[3] == "p":
                    deprel = "amod"  # participle as modifier
                else:
                    deprel = "acl"  # clause as modifier
            elif proiel_pos in ("Pr",):
                deprel = "acl:relcl"  # relative clause
            elif proiel_pos in ("Ma", "Mo"):
                deprel = "nummod"
            elif proiel_pos in ("Pp", "Pk", "Px", "Py", "Pi", "Pc"):
                deprel = "nmod"  # pronoun as modifier
            elif proiel_pos == "C-":
                deprel = "cc"
            elif proiel_pos in ("Df", "Dq"):
                deprel = "advmod"
            else:
                deprel = "nmod"

        # PROIEL "obl" - when POS is preposition, it should be "case"
        if relation == "obl" and proiel_pos == "R-":
            deprel = "case"

        # PROIEL "sub" - passive subject
        if relation == "sub" and proiel_pos == "V-":
            if morph and len(morph) >= 4 and morph[3] == "n":
                deprel = "csubj"  # infinitive as subject
            elif morph and len(morph) >= 4 and morph[3] == "p":
                deprel = "csubj"  # participle clause as subject

        # PROIEL "comp" - complement clauses
        if relation == "comp":
            if proiel_pos == "V-":
                deprel = "ccomp"
            else:
                deprel = "ccomp"

        # PROIEL "xadv" - open adverbial complement
        if relation == "xadv":
            if proiel_pos == "C-":
                deprel = "cc"
            elif proiel_pos == "R-":
                deprel = "case"
            else:
                deprel = "advcl"

        # PROIEL "xobj" - open objective complement
        if relation == "xobj":
            if proiel_pos == "R-":
                deprel = "case"
            else:
                deprel = "xcomp"

        # PROIEL "part" - partitive: nmod when head is nominal, obl when head is verbal
        if relation == "part":
            if head_id and head_id in old_to_new:
                head_idx = old_to_new[head_id] - 1  # 0-indexed into tokens list
                if head_idx < len(tokens):
                    head_proiel_pos = tokens[head_idx].get("part-of-speech", "")
                    head_upos = POS_MAP.get(head_proiel_pos, "X")
                    if head_upos in ("NOUN", "PROPN", "PRON", "NUM", "ADJ", "DET"):
                        deprel = "nmod"
                    else:
                        deprel = "obl"

        # Root handling
        if head == 0:
            deprel = "root"

        feat_str = feats_to_str(feats)
        parsed.append({
            "id": i, "form": form, "lemma": lemma, "upos": upos,
            "xpos": proiel_pos, "feats": feat_str, "head": head,
            "deprel": deprel,
        })

    # --- Tree restructuring: PP head inversion ---
    # In PROIEL, prepositions head their NP: noun -> prep (obl/adv/atr)
    # In UD, nouns head their PP: prep -> noun (case)
    # Find prepositions that have noun/pronoun dependents and invert.
    parsed = _restructure_pps(parsed)

    lines = []
    for p in parsed:
        lines.append(
            f"{p['id']}\t{p['form']}\t{p['lemma']}\t{p['upos']}\t"
            f"{p['xpos']}\t{p['feats']}\t{p['head']}\t{p['deprel']}\t_\t_"
        )

    return lines


def _restructure_pps(tokens):
    """Restructure prepositional phrases: make noun the head, prep the case dependent.

    In PROIEL, PPs are headed by the preposition:
        eaten(3) -> by(2, obl) -> dog(1, obl)  [dog depends on prep]

    In UD, PPs are headed by the noun:
        eaten(3) -> dog(1, obl) -> by(2, case)  [prep depends on noun]

    For each preposition that is already labeled 'case' (from our deprel mapping),
    it's already correct - skip it. For prepositions that HEAD nominal dependents
    (the PROIEL pattern), we need to invert.
    """
    n = len(tokens)
    if n == 0:
        return tokens

    # Build index: token id -> index
    id_to_idx = {t["id"]: i for i, t in enumerate(tokens)}

    # Find prepositions (ADP) that are NOT already case dependents
    # and that have nominal children (NOUN, PROPN, PRON, DET, NUM, ADJ)
    nominal_upos = {"NOUN", "PROPN", "PRON", "DET", "NUM", "ADJ"}

    for prep_idx, prep in enumerate(tokens):
        if prep["upos"] != "ADP":
            continue
        if prep["deprel"] == "case":
            continue  # already correct

        prep_id = prep["id"]

        # Find nominal children of this preposition
        nominal_children = []
        other_children = []
        for child_idx, child in enumerate(tokens):
            if child["head"] == prep_id:
                if child["upos"] in nominal_upos:
                    nominal_children.append(child_idx)
                else:
                    other_children.append(child_idx)

        if not nominal_children:
            continue

        # Check for coordination among children: if a CC (coordinating conj)
        # is among other_children, this is a coordinated PP. In UD, the first
        # conjunct should be the head.
        has_coord = any(
            tokens[ci]["upos"] == "CCONJ" or tokens[ci]["deprel"] == "cc"
            for ci in other_children
        )

        if has_coord:
            # For coordinated PPs, pick the FIRST (leftmost) nominal as head
            main_child_idx = nominal_children[0]
        else:
            # For simple PPs, pick the rightmost nominal (typical Greek order)
            main_child_idx = nominal_children[-1]
        main_child = tokens[main_child_idx]
        main_child_id = main_child["id"]

        # Invert: main child takes prep's position in the tree
        # 1. Main child gets prep's head and deprel
        main_child["head"] = prep["head"]
        main_child["deprel"] = prep["deprel"]

        # 2. Prep becomes case dependent of main child
        prep["head"] = main_child_id
        prep["deprel"] = "case"

        # 3. Other children of prep -> reparent to main child
        for ci in other_children:
            tokens[ci]["head"] = main_child_id

        # 4. Other nominal children of prep -> reparent to main child
        for ci in nominal_children:
            if ci != main_child_idx:
                tokens[ci]["head"] = main_child_id
                if has_coord:
                    # Coordinated PP: other nominals are conjuncts
                    tokens[ci]["deprel"] = "conj"
                elif tokens[ci]["deprel"] not in ("nmod", "appos", "conj"):
                    tokens[ci]["deprel"] = "nmod"

        # 5. Anything else in the sentence that pointed to the prep
        #    should now point to the main child instead
        for t in tokens:
            if t["head"] == prep_id and t["id"] != main_child_id and t["id"] != prep_id:
                if t is not prep:  # don't re-point the prep itself
                    t["head"] = main_child_id

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Convert DiGreC XML to CoNLL-U")
    parser.add_argument("--input", default="C:/Users/cisco/Documents/digrec/data/digrec.xml")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: data/DiGreC/)")
    parser.add_argument("--split", nargs=2, type=float, default=[0.8, 0.1],
                        help="Train/dev split ratios (rest is test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else Path(__file__).parent / "data" / "DiGreC"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {args.input}...")
    tree = ET.parse(args.input)
    root = tree.getroot()

    sentences = root.findall(".//sentence")
    print(f"Found {len(sentences)} sentences")

    # Convert all sentences
    converted = []
    skipped = 0
    for sent in sentences:
        lines = convert_sentence(sent)
        if lines:
            converted.append(lines)
        else:
            skipped += 1

    print(f"Converted: {len(converted)}, Skipped: {skipped}")

    # Split
    random.seed(args.seed)
    indices = list(range(len(converted)))
    random.shuffle(indices)

    n_train = int(len(converted) * args.split[0])
    n_dev = int(len(converted) * args.split[1])

    train_idx = indices[:n_train]
    dev_idx = indices[n_train:n_train + n_dev]
    test_idx = indices[n_train + n_dev:]

    def write_conllu(path, idx_list):
        with open(path, "w", encoding="utf-8") as f:
            for idx in idx_list:
                for line in converted[idx]:
                    f.write(line + "\n")
                f.write("\n")

    write_conllu(out_dir / "digrec-train.conllu", train_idx)
    write_conllu(out_dir / "digrec-dev.conllu", dev_idx)
    write_conllu(out_dir / "digrec-test.conllu", test_idx)

    total_tokens = sum(len(converted[i]) for i in range(len(converted)))
    print(f"\nSplit: {len(train_idx)} train, {len(dev_idx)} dev, {len(test_idx)} test")
    print(f"Total tokens: {total_tokens}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
