#!/usr/bin/env python3
"""One-shot Kaggle runner for reviewed JSON remote jobs."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


WORKING = Path("/kaggle/working")
CHECKOUT = WORKING / "rcq_moe_impl"
JOB_CHECKOUT = WORKING / "rcq_moe_jobs"
OUTPUTS_ROOT = WORKING / "outputs"
DEFAULT_JOB_REF = "kaggle-jobs"


def run(argv: list[str], *, cwd: Path | None = None) -> str:
    print("$", " ".join(shlex.quote(part) for part in argv), flush=True)
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, argv, result.stdout, result.stderr)
    return result.stdout.strip()


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for the one-shot remote worker.")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_sha(repo: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=repo)


def main() -> int:
    started = time.time()
    started_utc = utc_now()
    repo_url = required_env("RCQ_REPO_URL")
    job_ref = os.environ.get("RCQ_JOB_REF", DEFAULT_JOB_REF)
    job_path_value = required_env("RCQ_JOB_PATH")
    job_path = Path(job_path_value)

    if job_path.is_absolute() or ".." in job_path.parts:
        raise RuntimeError("RCQ_JOB_PATH must be a relative path inside the job checkout.")

    os.environ.setdefault("HF_HOME", str(WORKING / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(WORKING / "hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if CHECKOUT.exists():
        run(["rm", "-rf", str(CHECKOUT)])
    if JOB_CHECKOUT.exists():
        run(["rm", "-rf", str(JOB_CHECKOUT)])

    run(["git", "clone", repo_url, str(CHECKOUT)])
    run(["git", "clone", "--branch", job_ref, "--single-branch", repo_url, str(JOB_CHECKOUT)])

    spec_path = JOB_CHECKOUT / job_path
    import json

    raw_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    code_commit = raw_spec.get("code_commit")
    if not isinstance(code_commit, str) or not code_commit:
        raise RuntimeError("job spec must contain code_commit before checkout")

    run(["git", "fetch", "--depth", "1", "origin", code_commit], cwd=CHECKOUT)
    run(["git", "checkout", "--detach", code_commit], cwd=CHECKOUT)
    sys.path.insert(0, str(CHECKOUT))

    from rcq_moe.remote_jobs import (
        assert_output_dir_allowed,
        build_action_argv,
        load_job_spec,
        run_action,
        selected_env_snapshot,
        write_json,
    )

    job = load_job_spec(spec_path)
    output_dir = OUTPUTS_ROOT / job.job_id
    assert_output_dir_allowed(output_dir, outputs_root=OUTPUTS_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "job.json").write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "job_id": job.job_id,
        "action": job.action,
        "started_utc": started_utc,
        "ended_utc": None,
        "elapsed_sec": None,
        "status": "running",
        "repo_url": repo_url,
        "job_ref": job_ref,
        "job_path": str(job_path),
        "code_commit": job.code_commit,
        "checkout_commit": None,
        "argv": None,
        "env": selected_env_snapshot(),
        "returncode": None,
        "exception": None,
        "output_files": [],
    }
    write_json(output_dir / "run_manifest.json", manifest)

    try:
        manifest["checkout_commit"] = git_sha(CHECKOUT)
        manifest["argv"] = build_action_argv(job, output_dir=output_dir)
        write_json(output_dir / "run_manifest.json", manifest)

        result = run_action(job, repo_root=CHECKOUT, output_dir=output_dir, env=os.environ.copy())
        manifest["returncode"] = result.returncode
        manifest["status"] = "succeeded" if result.returncode == 0 else "failed"
        if result.returncode != 0:
            manifest["exception"] = f"action exited with returncode {result.returncode}"
    except Exception:
        manifest["status"] = "failed"
        manifest["exception"] = traceback.format_exc()
    finally:
        manifest["ended_utc"] = utc_now()
        manifest["elapsed_sec"] = time.time() - started
        if output_dir.exists():
            manifest["output_files"] = sorted(
                str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()
            )
        write_json(output_dir / "run_manifest.json", manifest)

    return 0 if manifest["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
