# Opla <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/languages/el.svg" width="28" alt="Greek"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/countries/cy.svg" width="28" alt="Cyprus"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/byzantine.svg" width="28" alt="Byzantine"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/ancient-greece.svg" width="28" alt="Ancient Greece">

<p align="center">
  <img src="opla.jpg" width="600" alt="Opla - tools spelled out in ancient metalworking tools">
</p>

GPU-optimized Greek POS tagger and dependency parser. **215x faster** than
[gr-nlp-toolkit](https://github.com/nlpaueb/gr-nlp-toolkit) on real-world
Greek text, with identical POS output and near-identical dependency parsing.
Supports Modern Greek, Ancient Greek, and Medieval Greek.

Opla (from Ancient Greek `ὅπλα`, "tools, equipment") is a drop-in replacement
for *gr-nlp-toolkit*'s POS and DP processors. It reuses *gr-nlp-toolkit*'s
trained weights for Modern Greek and adds Ancient Greek support via
custom-trained heads on [Ancient-Greek-BERT](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT).

## Installation

```bash
pip install -e .
```

Requires *PyTorch*, *Transformers*, and *huggingface-hub*. MG weights are
downloaded from [AUEB-NLP/gr-nlp-toolkit](https://huggingface.co/AUEB-NLP/gr-nlp-toolkit)
on first use. AG and Medieval weights are downloaded from
[ciscoriordan/opla](https://huggingface.co/ciscoriordan/opla).

Optional: install [Dilemma](https://github.com/ciscoriordan/dilemma) for
integrated lemmatization.

## Usage

```python
from opla import Opla

model = Opla(lang="el", device="cuda")    # Modern Greek
model = Opla(lang="grc", device="cuda")   # Ancient Greek

results = model.tag(["Ο Αχιλλέας πολεμά", "Η Ελένη φεύγει"])

for token in results[0]:
    print(token)
# {'form': 'ο', 'upos': 'DET', 'lemma': 'ο',
#  'feats': {'Number': 'Sing', 'Gender': 'Masc', 'PronType': 'Art',
#             'Definite': 'Def', 'Case': 'Nom'},
#  'head': 2, 'deprel': 'det'}
# ...
```

Pass any number of sentences. Opla handles batching internally.

### Automatic sentence segmentation

Pass unsegmented Greek text with `segment_text=True` to auto-split on
sentence-ending punctuation (`.` `;` `·` `!`), with abbreviation
awareness for both Modern and Ancient Greek:

```python
text = "Τι κάνεις; Καλά. Ο Αχιλλέας πολεμά!"
results = model.tag(text, segment_text=True)
# Returns 3 sentence results
```

The segmenter can also be used standalone:

```python
from opla.segment import segment
sentences = segment("π.χ. αυτό είναι μία πρόταση. Αυτή είναι άλλη.")
# ['π.χ. αυτό είναι μία πρόταση.', 'Αυτή είναι άλλη.']
```

### Options

```python
Opla(
    device="cuda",       # "cuda", "cpu", or None (auto-detect)
    lemmatize=True,      # include Dilemma lemmas in output
    max_subwords=2048,   # subword budget per batch (tune for VRAM)
    checkpoint="onnx",   # use ONNX weights if available (AG/med only)
)
```

## Why not gr-nlp-toolkit?

The trained BERT weights are good. The inference code is not.
We submitted [PR #29](https://github.com/nlpaueb/gr-nlp-toolkit/pull/29)
upstream (not merged). Opla goes further.

| Issue | *gr-nlp-toolkit* | Opla |
|-------|-----------------|------|
| BERT forward passes per sentence | **19** (POS loops over 17 features, calling BERT each time; DP calls BERT twice) | **2** (one per task) |
| Batching | `batch_size=1` hardcoded | Dynamic batching (~64-150 sentences) |
| BERT instances in VRAM | 2 identical copies (~880 MB wasted) | 2 distinct copies (needed - weights diverged during training) |
| `_` features in output | Emitted (e.g. `Case: _` on verbs) | Suppressed |
| Weight loading | `strict=False` (silent failures) | Validated on load |

### Benchmark

| | *gr-nlp-toolkit* | Opla | Speedup |
|---|---|---|---|
| **Full Iliad (24 books, 146K tokens)** | 4,193s (70 min) | **19.5s** | **215x** |
| Book 1 (611 sentences, 5,772 tokens) | 169.4s | 1.0s | 170x |
| BERT passes per sentence | 19 | 2 (el) / 1 (grc) | 9.5-19x |
| Batching | 1 sentence | ~64-150 sentences | 64x |
| Lemmatization | N/A | Batched via [Dilemma](https://github.com/ciscoriordan/dilemma) | |

### Accuracy vs *gr-nlp-toolkit*

Tested on all 611 sentences of Iliad Book 1 (Polylas MG translation,
5,772 tokens):

| Output | Match rate |
|--------|-----------|
| UPOS tags | **100.0%** (5,772/5,772) |
| Dependency relations | **99.8%** (5,760/5,772) |
| Dependency heads | **98.2%** (5,670/5,772) |

POS tagging is identical. The small head index differences come from how
padding affects biaffine attention scores in batched vs unbatched mode.
Dependency relation labels (what matters for downstream tasks like
redistribution bonding) are 99.8% identical.

## Architecture

For `lang="el"`, Opla loads *gr-nlp-toolkit*'s pre-trained weights from
[AUEB-NLP/gr-nlp-toolkit](https://huggingface.co/AUEB-NLP/gr-nlp-toolkit)
on HuggingFace and remaps them into a dual-backbone architecture on
[GreekBERT](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1):

```
Input sentences
    │
    ▼
Batched tokenization (GreekBERT tokenizer, uncased + deaccented)
    │  padding, attention masks, subword-to-word mapping
    │
    ├──▶ POS GreekBERT ──▶ 17 Linear heads ──▶ UPOS + morphological features
    │
    └──▶ DP GreekBERT  ──▶ Biaffine attention ──▶ dependency heads + relations
    │
    ▼
Decode: argmax, pos_properties filter, subword-to-word mapping
    │
    ▼
[{form, upos, lemma, feats, head, deprel}, ...]
```

Two separate GreekBERT instances are required because *gr-nlp-toolkit*
trained POS and DP with independent backbones that diverged during training.
Using the POS BERT for DP (or vice versa) degrades accuracy. This is still
a 9.5x reduction in BERT forward passes (2 vs 19).

For `lang="grc"` and `lang="med"`, Opla uses a single
[Ancient-Greek-BERT](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)
backbone with jointly trained POS+DP heads, requiring only one BERT forward
pass per batch.

### Dynamic batching

Sentences are accumulated until the total subword count reaches a threshold
(default 2,048), then flushed as a single padded batch. This adapts to
variable sentence lengths without exceeding VRAM limits. For typical Greek
text (~13 subwords per sentence), this means ~150 sentences per batch.

### Integrated lemmatization

When [Dilemma](https://github.com/ciscoriordan/dilemma) is installed, Opla
uses POS-aware lemmatization via `lemmatize_batch_pos()`, passing each
token's predicted UPOS tag to Dilemma for disambiguation (e.g., distinguishing
adverbial vs pronominal forms of the same surface word). The original polytonic
forms from the input text are preserved as `raw_form` and used for lookup,
since Dilemma's tables are keyed on polytonic Greek. Lookup table hits resolve
instantly; only unknown forms go through Dilemma's character-level transformer,
and those are batched too.

### ONNX inference

The model can be exported to ONNX via `export_onnx.py` for CPU-only
deployment with `onnxruntime` instead of PyTorch. For AG/med models
(single shared BERT), this produces a single combined file; for MG
(dual BERT), it produces separate POS and DP files.

```bash
# Export AG model to ONNX (~535 MB)
python export_onnx.py --lang grc --weights weights/grc/opla_grc.pt

# Output goes to weights/grc/onnx/opla_joint.onnx by default
```

To load the ONNX model at inference time, pass `checkpoint="onnx"`:

```python
model = Opla(lang="grc", checkpoint="onnx")
```

This requires `onnxruntime` (`pip install onnxruntime`). If ONNX weights
are not found or `onnxruntime` is not installed, Opla falls back to the
standard PyTorch checkpoint.

## Output format

Each token is a dict with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `form` | str | Surface word form (accent-stripped, lowercased) |
| `raw_form` | str | Original polytonic form from input text |
| `upos` | str | Universal POS tag (17 tags: VERB, NOUN, ADJ, DET, ...) |
| `lemma` | str | Lemma from Dilemma (if `lemmatize=True`) |
| `feats` | dict | Morphological features (Case, Gender, Number, Tense, ...) |
| `head` | int | Index of dependency head (0 = root) |
| `deprel` | str | Dependency relation (40 labels: nsubj, obj, det, root, ...) |

Features are filtered by `pos_properties` - only features valid for the
predicted UPOS tag are included. Underscore (`_`) values are suppressed.

## Files

```
opla/
    __init__.py      # Opla class, public API, dynamic batching
    model.py         # OplaModel - dual BERT + POS heads + DP biaffine heads
    weights.py       # Weight loading + key remapping from gr-nlp-toolkit
    tokenize.py      # Batched tokenization with subword-to-word mapping
    decode.py        # Decode logits to structured token dicts
    labels.py        # UPOS, morphological feature, and deprel label sets
    segment.py       # Greek sentence segmentation with abbreviation handling
    onnx_model.py    # ONNX runtime inference wrapper
train.py             # Train POS+DP heads on UD treebanks (grc, el, med)
convert_digrec.py    # Convert DiGreC PROIEL XML to CoNLL-U for med training
convert_gorman.py    # Convert Gorman AGDT XML to CoNLL-U for grc training
export_onnx.py       # Export model to ONNX format for CPU deployment
upload_weights.py    # Upload trained weights to HuggingFace
```

## Language coverage

### Currently supported: Modern Greek

The `el` model uses
[nlpaueb/bert-base-greek-uncased-v1](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1)
(GreekBERT) as its backbone - a 12-layer, 768-dim BERT-base model
pre-trained by AUEB-NLP on Greek Wikipedia, European Parliament proceedings,
and the OSCAR Common Crawl corpus. The tokenizer lowercases and strips
accents (NFD + remove combining marks), so all Greek text is normalized to
unaccented lowercase before entering the model.

POS and DP task heads come from *gr-nlp-toolkit*'s pre-trained weights,
which were fine-tuned on top of GreekBERT. Opla handles Katharevousa well
despite the MG training data because the BERT vocabulary covers the full
Greek script and most Katharevousa forms share stems with their MG
equivalents.

Tested on Iakovos Polylas's 1892 Iliad translation (Katharevousa-influenced
verse), which includes archaic verb forms, accusative -ν endings, polytonic
remnants, and poetic elisions not found in standard MG.

### Ancient Greek

The `grc` model uses
[pranaydeeps/Ancient-Greek-BERT](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)
as its backbone - initialized from GreekBERT and further pre-trained for 80
epochs on Ancient Greek corpora from the First1KGreek Project, Perseus
Digital Library, PROIEL Treebank, and Gorman's Treebanks. Same tokenizer
and preprocessing as GreekBERT (uncased, deaccented).

POS and DP task heads are jointly trained on 1.1M tokens from three
treebank sources, then fine-tuned with DiGreC data mixed in:

- [**UD_Ancient_Greek-Perseus**](https://universaldependencies.org/treebanks/grc_perseus/) - 203K tokens (Homer, Sophocles, Plato, Herodotus, Hesiod)
- [**UD_Ancient_Greek-PROIEL**](https://universaldependencies.org/treebanks/grc_proiel/) - 214K tokens (New Testament, Herodotus)
- [**Gorman's Greek Dependency Trees**](https://github.com/vgorman1/Greek-Dependency-Trees) - 692K tokens converted from AGDT XML to CoNLL-U via `convert_gorman.py`, with SCONJ/AUX mapping, elision handling, and 0.1% mismatch rate
- [**DiGreC**](https://proiel.github.io/digrec/) - 103K tokens (mixed fine-tuning, 3 epochs at lr=1e-5)

Because POS and DP are trained jointly from the start, the `grc` model uses
a single BERT backbone (1 forward pass per batch, vs 2 for `el`).

**Dev set accuracy (combined Perseus + PROIEL + Gorman):**

| Metric | Accuracy |
|--------|----------|
| UPOS | **96.8%** |
| DEPREL | **91.8%** |

AG uses an expanded label set: dual number, optative/subjunctive moods,
middle voice, locative case, and more.

To retrain from scratch: `python train.py --lang grc --epochs 10`
To fine-tune with DiGreC: `python train.py --resume weights/grc/opla_grc.pt --data data/UD_Ancient_Greek-Perseus/grc_perseus-ud-train.conllu data/UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu data/DiGreC/digrec-train.conllu --dev data/DiGreC/digrec-dev.conllu --epochs 3 --lr 1e-5`

### Medieval/Byzantine Greek

The `med` model uses the same
[Ancient-Greek-BERT](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)
backbone as `grc`, with POS+DP heads trained on the
[DiGreC treebank](https://proiel.github.io/digrec/) - 121K tokens of
Medieval Greek prose spanning Homer to early modern Greek (Thucydides,
Herodotus, Anna Komnene, Chronicle of Morea, and more).

DiGreC uses the PROIEL annotation scheme. Opla's `convert_digrec.py` script
converts it to UD-compatible CoNLL-U format with refined deprel mappings
(PROIEL `aux` -> UD `det`/`discourse`/`cc`/`mark`/`advmod` by POS;
`adv`+preposition -> `case`; `atr` -> `amod`/`nmod`/`det`/`nummod` by POS;
`part` -> `nmod`/`obl` by head POS). PP restructuring handles coordination
(first conjunct as head) and reparents dependents correctly.

**Dev set accuracy (586 sentences, 12 epochs):**

| Metric | Accuracy |
|--------|----------|
| UPOS | **96.4%** |
| Dependency heads | **75.4%** |
| Dependency relations | **90.2%** |
| Morphological features | 97-100% per feature |

Head accuracy is lower than AG because Medieval text has more complex
syntactic structures and the PROIEL->UD conversion requires tree
restructuring (PP head inversion, coordination handling).

To train: `python convert_digrec.py && python train.py --lang med --epochs 12`

### Multi-period API

```python
model = Opla(lang="el", device="cuda")    # Modern Greek
model = Opla(lang="grc", device="cuda")   # Ancient Greek
model = Opla(lang="med", device="cuda")   # Medieval/Byzantine Greek
```

Language codes: `el` (ISO 639-1), `grc` (ISO 639-2), `med` (Medieval
Greek). These codes are shared with
[Dilemma](https://github.com/ciscoriordan/dilemma), but the two tools
group `med` differently. Opla groups `med` with `grc` (shared BERT
backbone and task heads) because Medieval *syntax* - polytonic script,
full case system, optative mood - is closer to Ancient Greek.
Dilemma groups `med` with `el` for lemma lookup because Medieval
*morphology* (inflection patterns) is the direct ancestor of Modern
Greek. Each tool groups `med` with whichever period best serves its task.

## Credits

**Modern Greek backbone:**
[GreekBERT](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1) by the
[NLP Group at Athens University of Economics and Business](http://nlp.cs.aueb.gr/)
(AUEB-NLP). Pre-trained on Greek Wikipedia, European Parliament, and OSCAR.

**MG task heads:**
[gr-nlp-toolkit](https://github.com/nlpaueb/gr-nlp-toolkit) by AUEB-NLP.
POS and DP head architectures reproduced from *gr-nlp-toolkit*'s source code
to ensure weight compatibility.

**Ancient Greek backbone:**
[Ancient-Greek-BERT](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)
by Pranaydeep Singh (KU Leuven). Initialized from GreekBERT and further
pre-trained on First1KGreek, Perseus, PROIEL, and Gorman treebanks.

**AG training data:**
[UD_Ancient_Greek-Perseus](https://universaldependencies.org/treebanks/grc_perseus/)
and [UD_Ancient_Greek-PROIEL](https://universaldependencies.org/treebanks/grc_proiel/)
from the [Universal Dependencies](https://universaldependencies.org/) project.
[Greek Dependency Trees](https://github.com/vgorman1/Greek-Dependency-Trees)
by Vanessa Gorman (University of Nebraska-Lincoln).

**Medieval Greek training data:**
[DiGreC](https://proiel.github.io/digrec/) (Digitized Greek Corpus) by the
PROIEL project. 121K tokens of Medieval Greek with morphological and syntactic
annotation in the PROIEL scheme.

If you use Opla, please also cite:

```
Koutsikakis et al., "GREEK-BERT: The Greeks Visiting Sesame Street" (2020).
Toumazatos et al., "gr-nlp-toolkit: An open-source NLP toolkit for Modern Greek" (2024).
Singh et al., "A pilot study for BERT language modelling and morphological analysis for Ancient and Medieval Greek" (2021).
Eckhoff et al., "The PROIEL treebank family: a standard for early attestations of Indo-European languages" (2018).
Gorman, "Dependency Trees for Ancient Greek Prose" (2020).
```

## How to Cite

```
Francisco Riordan, "Opla: GPU-Optimized Greek POS Tagger and Dependency Parser" (2026).
https://github.com/ciscoriordan/opla
```

## Related projects

- [Dilemma](https://github.com/ciscoriordan/dilemma) - Greek lemmatizer (MG + AG + Medieval)
- [Dragoman](https://huggingface.co/ciscoriordan/dragoman) - Greek word alignment model

## Upcoming

- `pip install opla` - PyPI package for easy installation

## License

MIT

Flag icons by [svg-flags](https://github.com/ciscoriordan/svg-flags).
