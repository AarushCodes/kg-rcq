from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class TextFixtureConfig:
    max_docs: int = 256
    max_chars_per_doc: int = 2000
    eval_fraction: float = 0.2
    min_chars: int = 80


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

