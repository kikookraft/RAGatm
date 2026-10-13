"""Local LLM wrapper for the RAG pipeline.

This module loads the default Qwen model, builds a prompt from retrieved
source snippets, and generates a grounded answer while exposing visible
progress in the terminal.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterable, Sequence, cast

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

logger = logging.getLogger(__name__)


class ProgressTextIteratorStreamer(TextIteratorStreamer):
    """Text streamer that also advances a progress bar as tokens arrive."""

    def __init__(
        self,
        *args: Any,
        progress_bar: tqdm[Any],
        first_token_event: threading.Event,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.progress_bar = progress_bar
        self.first_token_event = first_token_event

    def put(self, value: torch.Tensor) -> None:
        """Forward tokens to the base streamer and update the progress bar."""
        if not self.first_token_event.is_set():
            self.first_token_event.set()
            self.progress_bar.set_postfix_str("streaming")
        self.progress_bar.update(1)
        super().put(value)


class LocalRAGModel:
    """Generate answers grounded in retrieved snippets."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 256,
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

    def build_prompt(self, question: str, sources: Iterable[str]) -> str:
        """Build the RAG prompt using only retrieved snippets as context."""
        cleaned_sources = [
            source.strip() for source in sources if source and source.strip()
        ]
        context = "\n\n---\n\n".join(cleaned_sources)
        if not context:
            context = "No relevant source context was provided."

        return (
            "You answer questions using only the provided source context. "
            "If the answer is not in the context, say so clearly. "
            "Return plain text only."
            f"\n\nQuestion: {question}\n\n"
            f"Relevant context:\n{context}"
        )

    def _build_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        """Create model inputs, preferring the chat template when available."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")

        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You answer using only the provided source context. "
                        "If the answer is not in the context, say so clearly. "
                        "Return plain text only."
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
        if not self.loaded:
            self._load_model()
            self.loaded = True
        """Generate a response for a prompt and return only the answer text."""
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
        with tqdm(
            total=self.max_new_tokens,
            desc="Generating",
            unit="tok",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            first_token_event = threading.Event()
            streamer = ProgressTextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                progress_bar=pbar,
                first_token_event=first_token_event,
            )
            generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs={
                    **inputs,
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": False,
                    "repetition_penalty": 1.1,
                    "no_repeat_ngram_size": 4,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "streamer": streamer,
                },
                daemon=True,
            )
            generation_thread.start()

            generated_chunks: list[str] = []
            while (
                not first_token_event.is_set()
                and generation_thread.is_alive()
            ):
                time.sleep(0.1)
            for chunk in streamer:
                if chunk:
                    generated_chunks.append(chunk)

            generation_thread.join()

        logger.info("Model output received.")
        answer = "".join(generated_chunks).strip()
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
