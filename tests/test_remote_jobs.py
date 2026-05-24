from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rcq_moe.remote_jobs import (
    JobSpecError,
    assert_output_dir_allowed,
    build_action_argv,
    load_job_spec,
    run_action,
    validate_job_spec,
)


def _base_payload(action: str = "control_plane_smoke") -> dict[str, object]:
    params: dict[str, object] = {
        "sequence_length": 64,
        "eval_batch_size": 1,
        "eval_batches": 1,
    }
    if action == "qwen36_fp_smoke":
        params.update(
            {
                "model_id": "Qwen/Qwen3.6-35B-A3B",
                "dtype": "bfloat16",
                "device_map": "auto",
                "trust_remote_code": False,
                "max_moe_modules": 25,
            }
        )
    return {
        "schema_version": 1,
        "job_id": "0001-control-plane-smoke" if action == "control_plane_smoke" else "0002-qwen36-fp-smoke",
        "action": action,
        "code_commit": "abcdef1234567890abcdef1234567890abcdef12",
        "created_utc": "2026-05-24T00:00:00Z",
        "reason": "unit test",
        "params": params,
    }


def test_valid_control_plane_job_builds_dry_run_argv() -> None:
    job = validate_job_spec(_base_payload())

    argv = build_action_argv(job, output_dir=Path("/tmp/out"), python_executable="python")

    assert argv == [
        "python",
        "scripts/qwen36_fp_smoke.py",
        "--output-dir",
        "/tmp/out",
        "--sequence-length",
        "64",
        "--eval-batch-size",
        "1",
        "--eval-batches",
        "1",
        "--dry-run",
    ]


def test_valid_qwen36_job_builds_fp_argv_with_defaults() -> None:
    payload = _base_payload("qwen36_fp_smoke")
    payload["params"] = {"trust_remote_code": True}
    job = validate_job_spec(payload)

    argv = build_action_argv(job, output_dir=Path("/tmp/out"), python_executable="python")

    assert "--dry-run" not in argv
    assert argv[:4] == ["python", "scripts/qwen36_fp_smoke.py", "--output-dir", "/tmp/out"]
    assert argv[argv.index("--model-id") + 1] == "Qwen/Qwen3.6-35B-A3B"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert "--trust-remote-code" in argv


def test_job_id_must_match_filename(tmp_path: Path) -> None:
    path = tmp_path / "different-id.json"
    path.write_text(json.dumps(_base_payload()), encoding="utf-8")

    with pytest.raises(JobSpecError, match="must match filename stem"):
        load_job_spec(path)


@pytest.mark.parametrize(
    "field",
    ["schema_version", "job_id", "action", "code_commit", "created_utc", "reason", "params"],
)
def test_missing_required_fields_rejected(field: str) -> None:
    payload = _base_payload()
    payload.pop(field)

    with pytest.raises(JobSpecError, match="missing required fields"):
        validate_job_spec(payload)


def test_unknown_top_level_field_rejected() -> None:
    payload = _base_payload()
    payload["extra"] = True

    with pytest.raises(JobSpecError, match="unknown top-level fields"):
        validate_job_spec(payload)


def test_unknown_action_rejected() -> None:
    payload = _base_payload()
    payload["action"] = "made_up_action"

    with pytest.raises(JobSpecError, match="unknown action"):
        validate_job_spec(payload)


def test_reserved_action_rejected() -> None:
    payload = _base_payload()
    payload["action"] = "debug_shell"

    with pytest.raises(JobSpecError, match="reserved"):
        validate_job_spec(payload)


def test_unknown_action_param_rejected() -> None:
    payload = _base_payload()
    payload["params"] = {"unexpected": 1}

    with pytest.raises(JobSpecError, match="unknown params"):
        validate_job_spec(payload)


def test_invalid_param_type_rejected() -> None:
    payload = _base_payload()
    payload["params"] = {"sequence_length": 0}

    with pytest.raises(JobSpecError, match="positive integer"):
        validate_job_spec(payload)


def test_invalid_commit_rejected() -> None:
    payload = _base_payload()
    payload["code_commit"] = "main"

    with pytest.raises(JobSpecError, match="hexadecimal"):
        validate_job_spec(payload)


def test_output_dir_must_stay_under_outputs_root(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    assert_output_dir_allowed(outputs_root / "job", outputs_root=outputs_root)

    with pytest.raises(JobSpecError, match="outside"):
        assert_output_dir_allowed(tmp_path / "other", outputs_root=outputs_root)


def test_run_action_executes_local_dry_run(tmp_path: Path) -> None:
    job = validate_job_spec(_base_payload())
    output_dir = tmp_path / "outputs" / job.job_id

    result = run_action(job, repo_root=Path.cwd(), output_dir=output_dir, python_executable=sys.executable)

    assert result.returncode == 0
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "module_structure.txt").read_text(encoding="utf-8") == "dry_run=true\n"
    assert (output_dir / "stdout.log").exists()
    assert (output_dir / "stderr.log").exists()


@pytest.mark.parametrize(
    "template_path",
    [
        Path("kaggle/remote_worker/job_templates/0001-control-plane-smoke.json"),
        Path("kaggle/remote_worker/job_templates/0002-qwen36-fp-smoke.json"),
    ],
)
def test_job_templates_validate_against_their_filenames(template_path: Path) -> None:
    job = load_job_spec(template_path)

    assert job.job_id == template_path.stem
