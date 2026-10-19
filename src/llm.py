"""Local LLM wrapper for the RAG pipeline.

This module loads the default Qwen model, builds a prompt from retrieved
source snippets, and generates a grounded answer while exposing visible
progress in the terminal.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Sequence, cast

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_LEADING_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:answer\s*[:\-]\s*|the answer is\s*[:\-]?\s*)",
    re.IGNORECASE,
)


class LocalRAGModel:
    """Generate answers grounded in retrieved snippets."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 1024,
    ) -> None:
        """Initialize the local tokenizer and model.

        Args:
            model_name: Hugging Face model identifier to load.
            max_new_tokens: Maximum number of output tokens to generate.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.loaded = False

    def _load_model(self) -> None:
        """Load the tokenizer and model with visible progress feedback."""
        logger.info(
            "Preparing to download/load %s from "
            "Hugging Face. This can take a while.",
            self.model_name,
        )
        logger.info("Loading tokenizer for %s", self.model_name)
        with tqdm(total=1, desc="Tokenizer", unit="step", leave=False) as pbar:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            pbar.update(1)
        logger.info("Tokenizer ready.")

        logger.info("Loading model %s", self.model_name)
        with tqdm(total=1, desc="Model", unit="step", leave=False) as pbar:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            pbar.update(1)
        logger.info("Model loaded successfully.")

    @staticmethod
    def _question_terms(question: str) -> set[str]:
        terms = {
            match.group(0).lower()
            for match in _WORD_RE.finditer(question)
        }
        return {
            term for term in terms
            if len(term) > 2 and term not in {"how", "what", "when", "where", "why", "which", "load"}
        }

    @classmethod
    def _select_relevant_text(cls, question: str, source: str) -> str:
        """Keep only the most question-relevant sentences from a source."""
        cleaned_source = source.strip()
        if not cleaned_source:
            return ""

        question_terms = cls._question_terms(question)
        if not question_terms:
            return cleaned_source

        selected_sentences: list[str] = []
        for sentence in _SENTENCE_RE.split(cleaned_source):
            normalized_sentence = sentence.strip()
            if not normalized_sentence:
                continue
            lowered_sentence = normalized_sentence.lower()
            if any(term in lowered_sentence for term in question_terms):
                selected_sentences.append(normalized_sentence)

        if selected_sentences:
            return " ".join(selected_sentences[:4])

        return cleaned_source

    def build_prompt(self, question: str, sources: Iterable[str]) -> str:
        """Build a concise RAG prompt from the retrieved snippets."""
        cleaned_sources = []
        for source in sources:
            if not source:
                continue
            relevant_text = self._select_relevant_text(question, source)
            if relevant_text.strip():
                cleaned_sources.append(relevant_text.strip())

        context = "\n\n---\n\n".join(cleaned_sources)
        if not context:
            context = "No relevant source context was provided."

        return (
            "You answer questions using only the provided source evidence. "
            "If the evidence contains the answer, state it directly and do "
            "not say the answer is missing. If the evidence truly does not "
            "contain the answer, say so clearly. Start directly with the "
            "substantive answer and return plain text only."
            f"\n\nQuestion: {question}\n\n"
            f"Relevant context:\n{context}\n"
        )

    @staticmethod
    def _strip_answer_prefix(answer: str) -> str:
        """Remove canned answer prefixes while preserving the content."""
        cleaned_answer = answer.strip()
        previous_answer = None
        while cleaned_answer and cleaned_answer != previous_answer:
            previous_answer = cleaned_answer
            cleaned_answer = _LEADING_ANSWER_PREFIX_RE.sub("", cleaned_answer)
        return cleaned_answer.strip()

    def _build_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        """Create model inputs, preferring the chat template when available."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")

        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You answer using only the provided source evidence. "
                        "If the evidence contains the answer, state it directly "
                        "and do not say the answer is missing. If the evidence "
                        "truly does not contain the answer, say so clearly. "
                        "Start directly with the substantive answer and return "
                        "plain text only."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            return cast(
                dict[str, torch.Tensor],
                self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
            )

        return cast(
            dict[str, torch.Tensor],
            self.tokenizer(prompt, return_tensors="pt"),
        )

    def generate(self, prompt: str) -> str:
        """Generate a response for a prompt and return only the answer text."""
        if not self.loaded:
            self._load_model()
            self.loaded = True

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        logger.info("Encoding prompt (%d chars)", len(prompt))
        inputs = self._build_inputs(prompt)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {
                key: value.to(self.device) for key, value in inputs.items()
            }

        logger.info(
            "Generating answer with max_new_tokens=%d",
            self.max_new_tokens,
        )
        input_length = inputs["input_ids"].shape[1]
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0, input_length:]
        answer = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        answer = self._strip_answer_prefix(answer)

        logger.info("Answer generated successfully.")
        return answer

    def answer(self, question: str, sources: Sequence[str]) -> str:
        """Answer a single question using the retrieved snippets."""
        prompt = self.build_prompt(question, sources)
        return self.generate(prompt)

    def answer_batch(
        self,
        questions: Sequence[
            tuple[str, Sequence[str]]
        ],
    ) -> list[str]:
        """Answer multiple questions and show progress with tqdm."""
        results: list[str] = []
        for index, (question, sources) in enumerate(
            tqdm(
                questions,
                desc="Answering questions",
                unit="q",
                leave=True,
            ),
            start=1,
        ):
            logger.info("Question %d/%d", index, len(questions))
            results.append(self.answer(question, sources))
        return results


llm = LocalRAGModel()


if __name__ == "__main__":
    example_question = "How do I configure the OpenAI-compatible server?"
    example_sources = [
        (
            "The OpenAI compatible server supports a --host option "
            "and a --port option."
        ),
        (
            "Use the API server entry point to expose "
            "OpenAI-compatible endpoints."
        ),
    ]
    print(llm.answer(example_question, example_sources))
