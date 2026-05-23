from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import torch


@dataclass(frozen=True)
class TextFixtureConfig:
    max_docs: int = 256
    max_chars_per_doc: int = 2000
    eval_fraction: float = 0.2
    min_chars: int = 80


@dataclass(frozen=True)
class TextBatchConfig:
    batch_size: int = 4
    sequence_length: int = 32
    max_batches: int = 1
    repeat_if_needed: bool = False


def normalize_text(text: str) -> str:
    """Normalize streamed web text into a compact deterministic fixture format."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def prepare_documents(raw_texts: Iterable[str], config: TextFixtureConfig) -> list[str]:
    docs: list[str] = []
    for raw in raw_texts:
        text = normalize_text(raw)
        if len(text) < config.min_chars:
            continue
        docs.append(text[: config.max_chars_per_doc].strip())
        if len(docs) >= config.max_docs:
            break
    return docs


def split_calib_eval(docs: list[str], eval_fraction: float) -> tuple[list[str], list[str]]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1.")
    if len(docs) < 2:
        raise ValueError("need at least two documents to split calibration and evaluation text.")
    eval_count = max(1, round(len(docs) * eval_fraction))
    eval_count = min(eval_count, len(docs) - 1)
    return docs[:-eval_count], docs[-eval_count:]


def write_text_fixture(docs: list[str], output_path: str | Path, *, title: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", ""]
    for idx, doc in enumerate(docs):
        body.append(f"--- document {idx} ---")
        body.append(doc)
        body.append("")
    output.write_text("\n".join(body).strip() + "\n", encoding="utf-8")


def read_text_fixture(input_path: str | Path) -> list[str]:
    """Read documents written by write_text_fixture, or treat plain text as one document."""
    text = Path(input_path).read_text(encoding="utf-8")
    docs: list[str] = []
    current: list[str] = []
    saw_marker = False

    def flush_current() -> None:
        normalized = normalize_text("\n".join(current))
        if normalized:
            docs.append(normalized)
        current.clear()

    for line in text.splitlines():
        is_marker = line.startswith("--- document ") and line.endswith(" ---")
        if is_marker:
            if saw_marker:
                flush_current()
            saw_marker = True
            continue
        if saw_marker:
            current.append(line)

    if saw_marker:
        flush_current()
        return docs

    normalized = normalize_text(text)
    return [normalized] if normalized else []


def encode_texts_as_toy_byte_tokens(texts: Iterable[str], *, vocab_size: int) -> list[int]:
    """Deterministic byte tokenizer for tiny random models.

    Token id 0 is reserved for padding/future masks; nonzero ids are byte values
    folded into the configured toy vocabulary.
    """
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2 for toy byte tokenization.")
    payload = "\n\n".join(normalize_text(text) for text in texts if normalize_text(text))
    byte_values = payload.encode("utf-8")
    return [(value % (vocab_size - 1)) + 1 for value in byte_values]


def texts_to_input_ids(texts: Iterable[str], *, vocab_size: int, config: TextBatchConfig) -> torch.Tensor:
    """Convert text documents into a fixed-shape input_ids tensor for tiny Qwen smoke tests."""
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if config.max_batches <= 0:
        raise ValueError("max_batches must be positive.")

    ids = encode_texts_as_toy_byte_tokens(texts, vocab_size=vocab_size)
    required = config.batch_size * config.sequence_length * config.max_batches
    if not ids:
        raise ValueError("cannot build input_ids from empty text.")
    if len(ids) < required:
        if not config.repeat_if_needed:
            raise ValueError(f"not enough toy tokens: have {len(ids)}, need {required}.")
        repeats = (required + len(ids) - 1) // len(ids)
        ids = (ids * repeats)[:required]
    else:
        ids = ids[:required]

    return torch.tensor(ids, dtype=torch.long).view(config.batch_size * config.max_batches, config.sequence_length)


def synthetic_fixture_documents() -> list[str]:
    """Small built-in offline fixture for tests and no-network smoke runs."""
    return [
        "Router coherent quantization keeps a shared low rank component in higher precision and quantizes only the residual expert weights.",
        "A calibration pass records selected experts, routing weights, and activations. The covariance matrix estimates which input directions matter.",
        "def affine_correction(y, yhat):\n    alpha = cov(y, yhat) / (var(yhat) + eps)\n    beta = mean(y) - alpha * mean(yhat)\n    return alpha, beta",
        "If W equals A times B plus R, then full rank shared factors can represent the matrix exactly and the residual becomes nearly zero.",
        "User: Why do we stream data?\nAssistant: Streaming avoids downloading a large web corpus while still giving realistic calibration text.",
        "Checklist: collect activations, decompose experts, rotate residuals, quantize signs, rescue sensitive blocks, fit routed correction.",
        "The evaluation split must use different text from calibration so the pipeline catches tokenization and batching mistakes.",
        "For a tiny random model, KL is a plumbing metric. It is not evidence that a compressed pretrained model will keep task quality.",
    ]


def load_hf_stream_texts(dataset: str, *, name: str | None, split: str, text_field: str) -> Iterable[str]:
    from datasets import load_dataset

    kwargs = {"path": dataset, "split": split, "streaming": True}
    if name:
        kwargs["name"] = name
    stream = load_dataset(**kwargs)
    for row in stream:
        value = row.get(text_field)
        if isinstance(value, str):
            yield value
