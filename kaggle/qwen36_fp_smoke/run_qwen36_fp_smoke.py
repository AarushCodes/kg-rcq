#!/usr/bin/env python3
"""Kaggle competition-attached runner for the Qwen3.6 FP smoke slice."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse


WORKING = Path("/kaggle/working")
CHECKOUT = WORKING / "rcq_moe_impl"
OUTPUT_DIR = WORKING / "outputs" / "qwen36_fp_smoke"
COMPETITION_DIR = Path("/kaggle/input/nvidia-nemotron-model-reasoning-challenge")


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    display_argv: list[str] | None = None,
) -> None:
    shown = display_argv if display_argv is not None else argv
    print("$", " ".join(shlex.quote(part) for part in shown), flush=True)
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is required. Add it as a Kaggle secret/env var; do not hardcode credentials in this kernel."
        )
    return value


def repo_url_with_optional_token(repo_url: str) -> tuple[str, str]:
    token = os.environ.get("RCQ_GIT_TOKEN")
    if not token:
        return repo_url, repo_url
    parsed = urlparse(repo_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("RCQ_GIT_TOKEN is only supported with an https RCQ_REPO_URL.")
    authed = urlunparse(parsed._replace(netloc=f"x-access-token:{token}@{parsed.netloc}"))
    redacted = urlunparse(parsed._replace(netloc=f"x-access-token:***@{parsed.netloc}"))
    return authed, redacted


def main() -> None:
    repo_url = required_env("RCQ_REPO_URL")
    commit_sha = os.environ.get("RCQ_COMMIT_SHA")
    model_id = os.environ.get("QWEN36_MODEL_ID", "Qwen/Qwen3.6-35B-A3B")

    os.environ.setdefault("HF_HOME", str(WORKING / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(WORKING / "hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"competition_dir_exists={COMPETITION_DIR.exists()}", flush=True)
    run(["nvidia-smi"])

    if CHECKOUT.exists():
        run(["rm", "-rf", str(CHECKOUT)])
    clone_url, display_clone_url = repo_url_with_optional_token(repo_url)
    run(
        ["git", "clone", "--depth", "1", clone_url, str(CHECKOUT)],
        display_argv=["git", "clone", "--depth", "1", display_clone_url, str(CHECKOUT)],
    )
    if commit_sha:
        run(["git", "fetch", "--depth", "1", "origin", commit_sha], cwd=CHECKOUT)
        run(["git", "checkout", "--detach", commit_sha], cwd=CHECKOUT)

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=CHECKOUT)

    script_args = [
        sys.executable,
        "scripts/qwen36_fp_smoke.py",
        "--model-id",
        model_id,
        "--output-dir",
        str(OUTPUT_DIR),
        "--sequence-length",
        os.environ.get("RCQ_FP_SMOKE_SEQUENCE_LENGTH", "256"),
        "--eval-batch-size",
        os.environ.get("RCQ_FP_SMOKE_EVAL_BATCH_SIZE", "1"),
        "--eval-batches",
        os.environ.get("RCQ_FP_SMOKE_EVAL_BATCHES", "1"),
    ]
    if os.environ.get("QWEN36_TRUST_REMOTE_CODE") == "1":
        script_args.append("--trust-remote-code")
    run(script_args, cwd=CHECKOUT)


if __name__ == "__main__":
    main()
