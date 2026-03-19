"""Opla: GPU-optimized Greek POS tagger + dependency parser.

Usage:
    from opla import Opla

    model = Opla(device="cuda")                    # MG (default)
    model = Opla(lang="grc", device="cuda")        # Ancient Greek
    results = model.tag(["Ο Αχιλλέας πολεμά"])
"""

from pathlib import Path

import torch
from transformers import AutoModel

from .model import OplaModel
from .labels import EL_POS_LABEL_COUNTS, EL_DP_LABEL_COUNT
from .weights import load_weights
from .tokenize import batch_tokenize
from .decode import decode_batch

__version__ = "0.2.0"

# Maximum subwords per dynamic batch before flushing to GPU
_DEFAULT_MAX_SUBWORDS = 2048

# BERT models per language
_BERT_MODELS = {
    "el": "nlpaueb/bert-base-greek-uncased-v1",
    "grc": "pranaydeeps/Ancient-Greek-BERT",
    "med": "pranaydeeps/Ancient-Greek-BERT",  # Medieval/Byzantine Greek
}

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


class Opla:
    """Greek POS tagger and dependency parser with integrated lemmatization.

    Supports Modern Greek (el) via gr-nlp-toolkit weights and Ancient Greek
    (grc) via custom-trained heads on UD Perseus + PROIEL treebanks.

    Args:
        lang: "el" (Modern Greek, default) or "grc" (Ancient Greek).
        device: "cuda", "cpu", or None (auto-detect).
        pos_path: Path to POS weights. None = auto-detect.
        dp_path: Path to DP weights. None = auto-detect (MG only).
        checkpoint: Path to a joint checkpoint (AG). Overrides pos/dp_path.
        max_subwords: Maximum subwords per batch before flushing.
        lemmatize: Whether to include lemmas in output (requires Dilemma).
    """

    def __init__(
        self,
        lang: str = "el",
        device: str | None = None,
        pos_path: str | None = None,
        dp_path: str | None = None,
        checkpoint: str | None = None,
        max_subwords: int = _DEFAULT_MAX_SUBWORDS,
        lemmatize: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_subwords = max_subwords
        self.lang = lang
        self._lemmatize = lemmatize
        self._lemmatizer = None

        if lang == "el":
            self._init_el(pos_path, dp_path)
        elif lang in ("grc", "med"):
            self._init_grc(checkpoint)
        else:
            raise ValueError(f"Unsupported language: {lang}. Use 'el', 'grc', or 'med'.")

        self.model.to(self.device)
        self.model.eval()

        if self._lemmatize:
            self._init_lemmatizer()

    def _init_el(self, pos_path, dp_path):
        """Initialize MG model with separate POS/DP BERTs (gr-nlp-toolkit weights)."""
        bert_name = _BERT_MODELS["el"]
        pos_bert = AutoModel.from_pretrained(bert_name)
        dp_bert = AutoModel.from_pretrained(bert_name)
        # Use MG-sized label counts for gr-nlp-toolkit weight compatibility
        self.model = OplaModel(
            pos_bert, dp_bert,
            feat_sizes=EL_POS_LABEL_COUNTS,
            num_deprels=EL_DP_LABEL_COUNT,
        )
        load_weights(self.model, pos_path=pos_path, dp_path=dp_path, device="cpu")

    def _init_grc(self, checkpoint):
        """Initialize AG/Medieval model with single BERT (jointly trained)."""
        if checkpoint is None:
            # Look for local weights first, then download from HuggingFace
            default = _WEIGHTS_DIR / self.lang / f"opla_{self.lang}.pt"
            if default.exists():
                checkpoint = str(default)
            else:
                try:
                    from huggingface_hub import hf_hub_download
                    checkpoint = hf_hub_download(
                        repo_id="ciscoriordan/opla",
                        filename=f"weights/{self.lang}/opla_{self.lang}.pt",
                    )
                except Exception:
                    raise FileNotFoundError(
                        f"Weights not found locally ({default}) or on HuggingFace. "
                        f"Train with: python train.py --lang {self.lang}"
                    )

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)

        bert_name = ckpt.get("bert_model", _BERT_MODELS["grc"])
        feat_sizes = ckpt.get("feat_sizes")
        num_deprels = ckpt.get("num_deprels")

        bert = AutoModel.from_pretrained(bert_name)
        self.model = OplaModel(
            bert,
            feat_sizes=feat_sizes,
            num_deprels=num_deprels,
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)

    def _init_lemmatizer(self):
        """Initialize Dilemma lemmatizer."""
        try:
            from dilemma import Dilemma
            dilemma_lang = "all"
            self._lemmatizer = Dilemma(lang=dilemma_lang, device="cpu")
        except ImportError:
            self._lemmatize = False

    def tag(self, sentences: list[str]) -> list[list[dict]]:
        """Tag a list of sentences, returning per-sentence token dicts.

        Handles dynamic batching internally - pass any number of sentences.

        Args:
            sentences: List of raw text strings (one sentence each).

        Returns:
            List of sentence results. Each sentence is a list of token dicts:
            {"form", "upos", "lemma", "feats", "head", "deprel"}
        """
        if not sentences:
            return []

        all_results = [None] * len(sentences)
        batch_indices = []
        batch_sentences = []
        est_subwords = 0

        for i, sent in enumerate(sentences):
            est = int(len(sent.split()) * 1.3) + 2
            if batch_sentences and est_subwords + est > self.max_subwords:
                results = self._tag_batch(batch_sentences)
                for idx, res in zip(batch_indices, results):
                    all_results[idx] = res
                batch_indices = []
                batch_sentences = []
                est_subwords = 0

            batch_indices.append(i)
            batch_sentences.append(sent)
            est_subwords += est

        if batch_sentences:
            results = self._tag_batch(batch_sentences)
            for idx, res in zip(batch_indices, results):
                all_results[idx] = res

        return all_results

    @torch.inference_mode()
    def _tag_batch(self, sentences: list[str]) -> list[list[dict]]:
        """Process a single batch through the model."""
        enc = batch_tokenize(sentences)

        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)

        pos_logits, arc_scores, rel_scores = self.model(input_ids, attention_mask)

        results = decode_batch(
            pos_logits, arc_scores, rel_scores,
            enc.word_masks, enc.subword2word, enc.word_forms,
        )

        if self._lemmatize and self._lemmatizer is not None:
            # Batch all forms across all sentences for one Dilemma call
            all_forms = [t["form"] for sent in results for t in sent]
            all_lemmas = self._lemmatizer.lemmatize_batch(all_forms)
            idx = 0
            for sent_tokens in results:
                for token in sent_tokens:
                    token["lemma"] = all_lemmas[idx]
                    idx += 1
        elif self._lemmatize:
            for sent_tokens in results:
                for token in sent_tokens:
                    token["lemma"] = token["form"]

        return results
