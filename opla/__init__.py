"""Opla: GPU-optimized Greek POS tagger + dependency parser.

Usage:
    from opla import Opla
    model = Opla(device="cuda")
    results = model.tag(["Ο Αχιλλέας πολεμά", "Η Ελένη φεύγει"])
"""

import torch
from transformers import AutoModel

from .model import OplaModel
from .weights import load_weights
from .tokenize import batch_tokenize
from .decode import decode_batch

__version__ = "0.1.0"

# Maximum subwords per dynamic batch before flushing to GPU
_DEFAULT_MAX_SUBWORDS = 2048


class Opla:
    """Greek POS tagger and dependency parser with integrated lemmatization.

    Fused BERT model with shared backbone - single forward pass for POS
    (17 feature heads) + dependency parsing (biaffine attention). Includes
    Dilemma lemmatizer for MG/AG lemmatization.

    Args:
        device: "cuda", "cpu", or None (auto-detect).
        pos_path: Path to POS weights. None = auto-download from HuggingFace.
        dp_path: Path to DP weights. None = auto-download from HuggingFace.
        max_subwords: Maximum subwords per batch before flushing.
        lemmatize: Whether to include lemmas in output (requires Dilemma).
    """

    def __init__(
        self,
        device: str | None = None,
        pos_path: str | None = None,
        dp_path: str | None = None,
        max_subwords: int = _DEFAULT_MAX_SUBWORDS,
        lemmatize: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_subwords = max_subwords
        self._lemmatize = lemmatize
        self._lemmatizer = None

        # Load dual BERT backbones + task heads
        pos_bert = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
        dp_bert = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
        self.model = OplaModel(pos_bert, dp_bert)
        load_weights(self.model, pos_path=pos_path, dp_path=dp_path, device="cpu")
        self.model.to(self.device)
        self.model.eval()

        # Lazy-load Dilemma
        if self._lemmatize:
            self._init_lemmatizer()

    def _init_lemmatizer(self):
        """Initialize Dilemma lemmatizer."""
        try:
            from dilemma import Dilemma
            self._lemmatizer = Dilemma(lang="all", device="cpu")
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

        # Dynamic batching: accumulate sentences until subword limit
        all_results = [None] * len(sentences)
        batch_indices = []
        batch_sentences = []
        est_subwords = 0

        for i, sent in enumerate(sentences):
            # Rough estimate: 1.3 subwords per whitespace word
            est = int(len(sent.split()) * 1.3) + 2  # +2 for [CLS]/[SEP]
            if batch_sentences and est_subwords + est > self.max_subwords:
                # Flush current batch
                results = self._tag_batch(batch_sentences)
                for idx, res in zip(batch_indices, results):
                    all_results[idx] = res
                batch_indices = []
                batch_sentences = []
                est_subwords = 0

            batch_indices.append(i)
            batch_sentences.append(sent)
            est_subwords += est

        # Flush remaining
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

        # Add lemmas via Dilemma
        if self._lemmatize and self._lemmatizer is not None:
            for sent_tokens in results:
                for token in sent_tokens:
                    token["lemma"] = self._lemmatizer.lemmatize(token["form"])
        elif self._lemmatize:
            # Dilemma not available, use form as fallback
            for sent_tokens in results:
                for token in sent_tokens:
                    token["lemma"] = token["form"]

        return results
