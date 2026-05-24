"""Validation and dispatch helpers for one-shot Kaggle remote jobs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
VALID_ACTIONS = {"control_plane_smoke", "qwen36_fp_smoke"}
RESERVED_ACTIONS = {"debug_shell", "one_layer_rcq", "poller", "github_result_push"}
REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "job_id",
    "action",
    "code_commit",
    "created_utc",
    "reason",
    "params",
}
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")

CONTROL_DEFAULTS: dict[str, Any] = {
    "sequence_length": 256,
    "eval_batch_size": 1,
    "eval_batches": 1,
}

QWEN36_FP_DEFAULTS: dict[str, Any] = {
    "model_id": "Qwen/Qwen3.6-35B-A3B",
    "sequence_length": 256,
    "eval_batch_size": 1,
    "eval_batches": 1,
    "dtype": "bfloat16",
    "device_map": "auto",
    "trust_remote_code": False,
    "max_moe_modules": 200,
}

ACTION_PARAM_DEFAULTS = {
    "control_plane_smoke": CONTROL_DEFAULTS,
    "qwen36_fp_smoke": QWEN36_FP_DEFAULTS,
}

DTYPE_VALUES = {"auto", "float32", "float16", "bfloat16"}
SECRET_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


class JobSpecError(ValueError):
    """Raised when a remote job spec fails validation."""


@dataclass(frozen=True)
class RemoteJob:
    schema_version: int
    job_id: str
    action: str
    code_commit: str
    created_utc: str
    reason: str
    params: dict[str, Any]


def _require_type(name: str, value: Any, expected_type: type) -> None:
    if not isinstance(value, expected_type):
        raise JobSpecError(f"{name} must be {expected_type.__name__}")


def _validate_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JobSpecError(f"{name} must be a positive integer")


def _validate_nonempty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise JobSpecError(f"{name} must be a nonempty string")


def _validate_created_utc(value: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JobSpecError("created_utc must be an ISO-like timestamp") from exc


def _validate_param_value(name: str, value: Any) -> None:
    if name in {"sequence_length", "eval_batch_size", "eval_batches", "max_moe_modules"}:
        _validate_positive_int(name, value)
    elif name in {"model_id", "device_map"}:
        _validate_nonempty_string(name, value)
    elif name == "dtype":
        if value not in DTYPE_VALUES:
            raise JobSpecError(f"dtype must be one of {sorted(DTYPE_VALUES)}")
    elif name == "trust_remote_code":
        if not isinstance(value, bool):
            raise JobSpecError("trust_remote_code must be bool")
    else:
        raise JobSpecError(f"unknown param {name!r}")


def validate_job_spec(payload: dict[str, Any], *, job_path: Path | None = None) -> RemoteJob:
    """Validate a decoded JSON job spec and return a normalized job."""

    if not isinstance(payload, dict):
        raise JobSpecError("job spec must be a JSON object")

    keys = set(payload)
    missing = REQUIRED_TOP_LEVEL_KEYS - keys
    extra = keys - REQUIRED_TOP_LEVEL_KEYS
    if missing:
        raise JobSpecError(f"missing required fields: {sorted(missing)}")
    if extra:
        raise JobSpecError(f"unknown top-level fields: {sorted(extra)}")

    if payload["schema_version"] != SCHEMA_VERSION:
        raise JobSpecError(f"schema_version must be {SCHEMA_VERSION}")

    job_id = payload["job_id"]
    action = payload["action"]
    code_commit = payload["code_commit"]
    created_utc = payload["created_utc"]
    reason = payload["reason"]
    params = payload["params"]

    _validate_nonempty_string("job_id", job_id)
    if not JOB_ID_RE.fullmatch(job_id):
        raise JobSpecError("job_id must be filesystem-safe")
    if job_path is not None and job_path.stem != job_id:
        raise JobSpecError(f"job_id {job_id!r} must match filename stem {job_path.stem!r}")

    _validate_nonempty_string("action", action)
    if action in RESERVED_ACTIONS:
        raise JobSpecError(f"action {action!r} is reserved for a future slice")
    if action not in VALID_ACTIONS:
        raise JobSpecError(f"unknown action {action!r}")

    _validate_nonempty_string("code_commit", code_commit)
    if not COMMIT_RE.fullmatch(code_commit):
        raise JobSpecError("code_commit must be a 7-40 character hexadecimal git SHA")

    _validate_nonempty_string("created_utc", created_utc)
    _validate_created_utc(created_utc)
    _validate_nonempty_string("reason", reason)
    _require_type("params", params, dict)

    allowed_params = set(ACTION_PARAM_DEFAULTS[action])
    extra_params = set(params) - allowed_params
    if extra_params:
        raise JobSpecError(f"unknown params for {action}: {sorted(extra_params)}")
    for name, value in params.items():
        _validate_param_value(name, value)

    normalized_params = dict(ACTION_PARAM_DEFAULTS[action])
    normalized_params.update(params)

    return RemoteJob(
        schema_version=SCHEMA_VERSION,
        job_id=job_id,
        action=action,
        code_commit=code_commit,
        created_utc=created_utc,
        reason=reason,
        params=normalized_params,
    )


def load_job_spec(path: Path) -> RemoteJob:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_job_spec(payload, job_path=path)


def build_action_argv(
    job: RemoteJob,
    *,
    output_dir: Path,
    python_executable: str = sys.executable,
    script_path: Path = Path("scripts/qwen36_fp_smoke.py"),
) -> list[str]:
    """Build the exact argv for an implemented named action."""

    params = job.params
    argv = [
        python_executable,
        str(script_path),
        "--output-dir",
        str(output_dir),
        "--sequence-length",
        str(params["sequence_length"]),
        "--eval-batch-size",
        str(params["eval_batch_size"]),
        "--eval-batches",
        str(params["eval_batches"]),
    ]
    if job.action == "control_plane_smoke":
        return argv + ["--dry-run"]
    if job.action == "qwen36_fp_smoke":
        argv.extend(
            [
                "--model-id",
                str(params["model_id"]),
                "--dtype",
                str(params["dtype"]),
                "--device-map",
                str(params["device_map"]),
                "--max-moe-modules",
                str(params["max_moe_modules"]),
            ]
        )
        if params["trust_remote_code"]:
            argv.append("--trust-remote-code")
        return argv
    raise JobSpecError(f"unknown action {job.action!r}")


def assert_output_dir_allowed(output_dir: Path, *, outputs_root: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_root = outputs_root.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as exc:
        raise JobSpecError(f"output_dir {output_dir} is outside {outputs_root}") from exc


def selected_env_snapshot(env: dict[str, str] | None = None) -> dict[str, Any]:
    source = os.environ if env is None else env
    return {
        "HF_HOME": source.get("HF_HOME"),
        "TRANSFORMERS_CACHE": source.get("TRANSFORMERS_CACHE"),
        "TOKENIZERS_PARALLELISM": source.get("TOKENIZERS_PARALLELISM"),
        "secret_env_present": {name: bool(source.get(name)) for name in SECRET_ENV_NAMES},
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_action(
    job: RemoteJob,
    *,
    repo_root: Path,
    output_dir: Path,
    env: dict[str, str] | None = None,
    python_executable: str = sys.executable,
) -> subprocess.CompletedProcess[str]:
    """Run a validated job action and capture stdout/stderr in the output dir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_action_argv(job, output_dir=output_dir, python_executable=python_executable)
    result = subprocess.run(argv, cwd=repo_root, env=env, text=True, capture_output=True, check=False)
    (output_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
    return result
