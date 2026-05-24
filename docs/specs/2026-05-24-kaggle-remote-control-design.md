# Kaggle Remote Control Design

Date: 2026-05-24

## Purpose

Define a peer-reviewable one-shot Kaggle control plane for remote GPU smoke
runs. The immediate goal is to make the public GitHub repo usable from Kaggle
without moving large model files to the local machine, while preserving exact
commit pinning, small output artifacts, and clean rollback points.

This design covers the remote-control plumbing only. It does not add
quantization of pretrained Qwen3.6 weights, fused kernels, result-branch writes,
long-running polling, or arbitrary remote shell execution.

## Chosen Approach

Use a small JSON action framework driven by a separate `kaggle-jobs` branch.

- `main` contains source code, tests, runner implementation, docs, and committed
  design specs.
- `kaggle-jobs` contains JSON job specs only.
- Each job spec pins the exact `main` commit to execute through `code_commit`.
- Kaggle runs one selected job per notebook/script invocation.
- Results remain in Kaggle output artifacts. Selected small summaries are
  committed locally after review.

This is a hybrid remote-control path: formal runs use named, committed actions,
while future debug and polling capabilities are reserved for later slices.

## Alternatives Considered

### Minimal One-Off Runner

Hardcode the two immediate Kaggle commands directly in a notebook/script.

This has the lowest implementation cost, but it would likely need restructuring
for the next slice because one-layer RCQ conversion needs a third action and
more run metadata.

### Full Remote Worker Framework

Implement polling, retries, debug shell, job locks, result pushing, and richer
manifests immediately.

This is more convenient later, but it adds unnecessary complexity before the
first real FP-only pretrained smoke result exists.

## Branch Model

The repo uses two long-lived branches for this workflow:

- `main`: implementation, tests, docs, Kaggle worker code, and reproducible
  scripts.
- `kaggle-jobs`: one-shot JSON job specs.

The Kaggle worker clones the public GitHub repo, fetches the job branch, reads a
selected job spec, then checks out `main` at the pinned `code_commit` before
running the action.

No GitHub write token is required on Kaggle for the initial workflow.

## Job Spec Schema

Job specs are JSON files under `jobs/` on the `kaggle-jobs` branch. The file
stem must match `job_id`.

Example:

```json
{
  "schema_version": 1,
  "job_id": "0001-control-plane-smoke",
  "action": "control_plane_smoke",
  "code_commit": "abcdef1234567890abcdef1234567890abcdef12",
  "created_utc": "2026-05-24T00:00:00Z",
  "reason": "Verify public GitHub clone, pinned checkout, GPU visibility, and dry-run output.",
  "params": {
    "sequence_length": 256,
    "eval_batch_size": 1,
    "eval_batches": 1
  }
}
```

Required top-level fields:

- `schema_version`: must be `1`.
- `job_id`: filesystem-safe ID matching the filename stem.
- `action`: implemented named action.
- `code_commit`: required git commit SHA resolvable from `main`.
- `created_utc`: ISO-like UTC timestamp for review context.
- `reason`: human-readable explanation of why the job exists.
- `params`: action-specific object.

Unknown top-level fields fail validation. Unknown action params fail
validation.

## Initial Actions

### `control_plane_smoke`

Purpose: verify the remote-control path without loading model weights.

Behavior:

- Verify Kaggle environment and GPU visibility.
- Verify public repo clone.
- Verify pinned commit checkout.
- Run `scripts/qwen36_fp_smoke.py --dry-run`.
- Write only small JSON/text outputs under the job output directory.

Allowed params:

- `sequence_length`
- `eval_batch_size`
- `eval_batches`

### `qwen36_fp_smoke`

Purpose: run the first FP-only pretrained Qwen3.6 smoke on Kaggle.

Behavior:

- Load tokenizer and model for `Qwen/Qwen3.6-35B-A3B` by default.
- Inspect sparse MoE/router/expert module structure.
- Run a tiny next-token loss pass.
- Write metadata, module structure text, and FP metrics JSON.
- Keep model/cache files on Kaggle and outside committed artifacts.

Allowed params:

- `model_id`, default `Qwen/Qwen3.6-35B-A3B`
- `sequence_length`
- `eval_batch_size`
- `eval_batches`
- `dtype`, default `bfloat16`
- `device_map`, default `auto`
- `trust_remote_code`, default `false`
- `max_moe_modules`, default `200`

## Reserved Future Actions

The worker should reject these actions until their slices are explicitly
designed and implemented:

- `debug_shell`
- `one_layer_rcq`
- long-running poller mode
- GitHub result pushing

Reserving the names makes intent clear without adding unreviewed command
freedom in the first worker slice.

## Components

### Kaggle Entrypoint

Planned path: `kaggle/remote_worker/run_job.py`.

Responsibilities:

- Read `RCQ_REPO_URL`, `RCQ_JOB_REF`, and `RCQ_JOB_PATH`.
- Clone the public repo.
- Fetch `kaggle-jobs`.
- Load the selected JSON job spec.
- Check out `code_commit`.
- Dispatch the action.
- Write `job.json`, `run_manifest.json`, logs, and action outputs.

### Local Validation And Dispatch Module

Planned path: `rcq_moe/remote_jobs.py`.

Responsibilities:

- Validate JSON specs.
- Enforce exact action and param allowlists.
- Construct command argv for each action.
- Keep logic unit-testable without Kaggle, GPU, network, or model weights.

### Tests

Planned path: `tests/test_remote_jobs.py`.

Coverage:

- Valid control-plane job accepted.
- Valid FP smoke job accepted.
- Missing required fields rejected.
- Unknown actions rejected.
- Unknown params rejected.
- Filename/job ID mismatch rejected.
- Reserved future actions rejected.
- Dry-run command construction remains stable.

## Data Flow

1. Make the repo public on GitHub.
2. Commit worker code on `main`.
3. Create a JSON job spec on `kaggle-jobs`, pinned to a `main` commit.
4. Start one Kaggle notebook/script invocation with repo URL, job branch/ref,
   and job path.
5. Kaggle executes exactly one job.
6. Kaggle writes small artifacts under `/kaggle/working/outputs/<job_id>/`.
7. Local workflow inspects logs and JSON/text outputs.
8. Successful slice summaries are committed locally into `state.md` and
   `STRUCTURE.MD`.

## Output Layout

Each job writes to:

```text
/kaggle/working/outputs/<job_id>/
```

Expected files:

- `job.json`: exact job spec copied from `kaggle-jobs`.
- `run_manifest.json`: worker metadata, timing, selected environment details,
  git SHA, action, argv, exit status, and output file list.
- action-specific outputs, such as:
  - `metadata.json`
  - `module_structure.txt`
  - `fp_metrics.json`
  - stdout/stderr logs when applicable

Large model files, tokenizer caches, and Hugging Face cache directories are not
copied into job outputs.

## Error Handling

The worker fails closed.

- Missing required fields fail before any model load.
- Unknown actions fail.
- Unknown params fail.
- `job_id`/filename mismatch fails.
- Missing or unresolvable `code_commit` fails.
- Output paths must stay under `/kaggle/working/outputs/<job_id>`.
- Runtime failures still write `run_manifest.json` with exception text when
  possible.
- `debug_shell` is rejected until a later approved slice implements it.

## Secrets

Initial workflow uses no GitHub token on Kaggle because the repo is public.

Allowed secrets:

- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`, only if model access requires it.

Disallowed in the initial workflow:

- GitHub write token.
- Committed credentials.
- Printing secret values in logs.

## Testing Plan

Local verification before Kaggle:

- Unit-test job validation and dispatch.
- Run local dry-run action without GPU, network, or pretrained weights.
- Keep `uv run pytest -q` green.

Kaggle verification:

1. Run `control_plane_smoke`.
2. Inspect artifacts and logs.
3. Commit selected state summary.
4. Run `qwen36_fp_smoke`.
5. Inspect only small JSON/text outputs.
6. Commit selected state summary.

## Slice Plan

### Slice 1: Design

Write and commit this design spec. No worker code is added in this slice.

### Slice 2: Worker

Add the JSON validation/dispatch module, one-shot Kaggle runner, tests, and
sample job templates. Validate locally with dry-run. Update `state.md` and
`STRUCTURE.MD`.

### Slice 3: Kaggle Control-Plane Smoke

Run `control_plane_smoke` on Kaggle. Do not load Qwen weights. Inspect small
artifacts and commit selected summaries.

### Slice 4: Kaggle FP-Only Qwen3.6 Smoke

Run `qwen36_fp_smoke` on Kaggle. Load pretrained weights only on Kaggle. Inspect
and summarize small outputs.

### Later Slices

Add `debug_shell` only if needed. Add bounded polling only if repeated one-shot
invocation becomes a bottleneck. Add one-layer RCQ conversion after the FP-only
pretrained smoke path is proven.
