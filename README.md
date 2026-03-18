# Opla <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/languages/el.svg" width="28" alt="Greek"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/countries/cy.svg" width="28" alt="Cyprus"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/byzantine.svg" width="28" alt="Byzantine"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/ancient-greece.svg" width="28" alt="Ancient Greece">

GPU-optimized Greek POS tagger and dependency parser. **117x faster** than
[gr-nlp-toolkit](https://github.com/nlpaueb/gr-nlp-toolkit) on real-world
Greek text, with identical POS output and near-identical dependency parsing.

Opla (from Ancient Greek ὅπλα, "tools, equipment") is a drop-in replacement
for gr-nlp-toolkit's POS and DP processors. It reuses gr-nlp-toolkit's
trained weights with no retraining required.

## Installation

```bash
pip install -e .
```

Requires PyTorch, Transformers, and huggingface-hub. Weights are downloaded
automatically from HuggingFace on first use.

Optional: install [Dilemma](https://github.com/ciscoriordan/dilemma) for
integrated lemmatization.

## Usage

```python
from opla import Opla

model = Opla(device="cuda")
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

### Options

```python
Opla(
    device="cuda",       # "cuda", "cpu", or None (auto-detect)
    lemmatize=True,      # include Dilemma lemmas in output
    max_subwords=2048,   # subword budget per batch (tune for VRAM)
)
```

## Why not gr-nlp-toolkit?

Opla exists because gr-nlp-toolkit has critical performance bugs that make it
orders of magnitude slower than it needs to be. The underlying BERT model and
trained task heads are good - the inference code around them is not.

We submitted a fix for the most egregious bug as
[PR #29](https://github.com/nlpaueb/gr-nlp-toolkit/pull/29) to the upstream
repo. As of writing, it has not been merged. Opla goes further than that PR
by adding true batching, fusing the model, and eliminating the remaining
inefficiencies.

### Bug 1: 19 redundant BERT forward passes per sentence

gr-nlp-toolkit's POS processor loops over 17 morphological features and runs
a full BERT forward pass for each one, even though the model's own `forward()`
method already returns all 17 features in a single pass. The processor calls
BERT, extracts one feature, throws away the other 16, then calls BERT again
for the next feature. This is in
[`processors/pos.py` lines 71-76](https://github.com/nlpaueb/gr-nlp-toolkit/blob/main/gr_nlp_toolkit/processors/pos.py):

```python
for feat in self.feat_to_I2L.keys():       # 17 iterations
    output = self._model(input_ids, ...)    # full BERT pass each time
    predictions[feat] = argmax(output[feat])
```

The DP processor does the same thing, running BERT twice (once for heads, once
for dependency relations), even though both are computed in a single forward
pass.

Total: **17 + 2 = 19 BERT forward passes per sentence.** Opla does it in 2
(one per task, since POS and DP were trained with separate BERT backbones).

### Bug 2: No batching (hardcoded batch_size=1)

gr-nlp-toolkit's `DatasetImpl.__len__` returns 1 unconditionally
([`domain/dataset.py` line 18](https://github.com/nlpaueb/gr-nlp-toolkit/blob/main/gr_nlp_toolkit/domain/dataset.py)),
and the DataLoader is created with no batch size parameter
([`processors/tokenizer.py` line 150](https://github.com/nlpaueb/gr-nlp-toolkit/blob/main/gr_nlp_toolkit/processors/tokenizer.py)).
Every sentence is processed in complete isolation, with no padding or
parallelism. On a GPU that can handle 64 sentences in parallel, this wastes
~98% of available compute.

### Bug 3: Two duplicate BERT instances

The POS and DP processors each independently call
`AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')`, loading two
separate copies of the same 110M-parameter model into VRAM (~880 MB total).
They never share weights or coordinate. In gr-nlp-toolkit's case this is pure
waste since both are initialized from the same checkpoint. In Opla's case, the
two instances are necessary because POS and DP were trained separately and
their BERT weights have diverged, but at least they're loaded intentionally.

### Bug 4: Underscore features emitted to output

The POS processor emits morphological features with value `_` (meaning "not
applicable") into the token's `feats` dict. For example, a VERB gets
`{'Case': '_', 'Gender': '_', ...}`. These are noise - the `pos_properties`
table already defines which features are valid per UPOS tag, but the
processor only uses it to decide whether to *add* a feature, not to filter
out `_` values. Downstream code has to check for and ignore these.
Opla suppresses `_` values at decode time.

### Bug 5: No `strict=True` on weight loading

Both POS and DP processors load pre-trained weights with `strict=False`
([`processors/pos.py` line 54](https://github.com/nlpaueb/gr-nlp-toolkit/blob/main/gr_nlp_toolkit/processors/pos.py),
[`processors/dp.py` line 52](https://github.com/nlpaueb/gr-nlp-toolkit/blob/main/gr_nlp_toolkit/processors/dp.py)).
This silently ignores missing or unexpected keys, meaning corrupted or
mismatched checkpoints load without error. Opla validates all keys on load
and raises immediately if anything is wrong.

### Benchmark: Iliad Book 1 (611 sentences, 5,772 tokens)

| | gr-nlp-toolkit | Opla | Speedup |
|---|---|---|---|
| Time | 169.4s | 1.4s | **117x** |
| Throughput | 3.4 sent/s | 436 sent/s | |
| BERT passes | 19/sentence | 2/sentence | 9.5x |
| Batching | 1 sentence | ~64 sentences | 64x |
| VRAM (BERT) | ~880 MB (wasted dup) | ~880 MB (needed) | |

### Accuracy vs gr-nlp-toolkit

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

Opla loads gr-nlp-toolkit's pre-trained weights from
[AUEB-NLP/gr-nlp-toolkit](https://huggingface.co/AUEB-NLP/gr-nlp-toolkit)
on HuggingFace and remaps them into a clean dual-backbone architecture:

```
Input sentences
    │
    ▼
Batched tokenization (HuggingFace BERT tokenizer)
    │  padding, attention masks, subword-to-word mapping
    │
    ├──▶ POS BERT backbone ──▶ 17 Linear heads ──▶ UPOS + morphological features
    │
    └──▶ DP BERT backbone ──▶ Biaffine attention ──▶ dependency heads + relations
    │
    ▼
Decode: argmax, pos_properties filter, subword-to-word mapping
    │
    ▼
[{form, upos, lemma, feats, head, deprel}, ...]
```

Two separate BERT instances are required because gr-nlp-toolkit trained POS
and DP with independent BERT backbones that diverged during training. Using
the POS BERT for DP (or vice versa) degrades accuracy. This is still a 9.5x
reduction in BERT forward passes (2 vs 19).

### Dynamic batching

Sentences are accumulated until the total subword count reaches a threshold
(default 2,048), then flushed as a single padded batch. This adapts to
variable sentence lengths without exceeding VRAM limits. For typical Greek
text (~13 subwords per sentence), this means ~150 sentences per batch.

### Integrated lemmatization

When [Dilemma](https://github.com/ciscoriordan/dilemma) is installed, Opla
calls it on each token after POS decoding, returning lemmas alongside POS and
DP output. This matches the output format expected by downstream tools like
`tag_polylas.py` in the
[iliad-align](https://github.com/ciscoriordan/iliad-align) pipeline.

## Output format

Each token is a dict with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `form` | str | Surface word form (accent-stripped, lowercased) |
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
```

## Language coverage

### Currently supported: Modern Greek

Opla currently supports **Modern Greek (MG)**, including the Katharevousa
register used in 19th-century literary translations. The underlying model was
trained by AUEB-NLP on contemporary MG corpora, but handles Katharevousa well
since the BERT backbone shares vocabulary and script with learned/archaic MG
forms.

Tested on Iakovos Polylas's 1892 Iliad translation (Katharevousa-influenced
verse), which includes archaic verb forms, accusative -ν endings, polytonic
remnants, and poetic elisions not found in standard MG.

### Planned: Ancient Greek

Ancient Greek support will use the same BERT backbone fine-tuned with AG task
heads trained on:

- **Perseus AGDT** (Ancient Greek Dependency Treebank) - 112K hand-annotated
  tokens covering the full Iliad, with POS, morphology, and dependency parsing
- **PROIEL Treebank** - ~200K tokens of prose (New Testament, Herodotus)
- **Gorman Treebanks** - additional AG prose and drama

AG requires an expanded label set (dual number, optative/subjunctive moods,
middle voice, dative case is productive rather than vestigial). The POS and DP
heads will be trained separately from the MG heads.

### Planned: Medieval/Byzantine Greek

Medieval Greek sits between AG and MG. Training data is limited, but the
[DiGreC treebank](https://cid.ulster.ac.uk/) (56K tokens spanning Homer to
early modern Greek) provides annotated Medieval/Byzantine material. A model
trained on both AG and MG endpoints should handle the transitional period
reasonably, with DiGreC for fine-tuning.

### Multi-period API (future)

```python
model = Opla(lang="el", device="cuda")     # Modern Greek (current)
model = Opla(lang="grc", device="cuda")   # Ancient Greek
model = Opla(lang="all", device="cuda")   # auto-detect or shared heads
```

This mirrors [Dilemma](https://github.com/ciscoriordan/dilemma)'s
`lang="all"` / `"el"` / `"grc"` interface.

## Credits

Opla uses pre-trained model weights from
[gr-nlp-toolkit](https://github.com/nlpaueb/gr-nlp-toolkit) by the
[NLP Group at Athens University of Economics and Business](http://nlp.cs.aueb.gr/)
(AUEB-NLP). The BERT backbone is
[nlpaueb/bert-base-greek-uncased-v1](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1).
The POS and DP task head architectures are reproduced from gr-nlp-toolkit's
source code to ensure weight compatibility.

If you use Opla, please also cite gr-nlp-toolkit:

```
Toumazatos et al., "gr-nlp-toolkit: An open-source NLP toolkit for Modern Greek" (2024).
```

## Related projects

- [Dilemma](https://github.com/ciscoriordan/dilemma) - Greek lemmatizer (MG + AG + Medieval)
- [Dragoman](https://huggingface.co/ciscoriordan/dragoman) - Greek word alignment model
- [iliad-align](https://github.com/ciscoriordan/iliad-align) - Iliad parallel reader alignment pipeline

## License

MIT

Flag icons by [svg-flags](https://github.com/ciscoriordan/svg-flags).
