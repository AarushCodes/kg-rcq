# MI300X Remote Workflow Design

Date: 2026-05-25

## Purpose

Define a peer-reviewable workflow for running the pretrained-compatible
RCQ-MoE smoke path on a single AMD MI300X remote machine. The immediate goal is
to replace the active Kaggle validation path with normal SSH-based remote
execution while preserving exact commit pinning, small reviewed outputs, clean
rollback points, and no local transfer of large model files.

This design covers remote execution discipline only. It does not add pretrained
quantization, fused kernels, production packing, multi-GPU sharding, or any
unreviewed remote shell framework.

## Chosen Approach

Use the MI300X host as a pinned execution target, not as an interactive notebook
source of truth.

- Local `main` remains the reviewed implementation source.
- Each remote run starts from an exact local git commit that is pushed to the
  GitHub remote.
- The MI300X checkout runs the exact commit SHA selected for the slice.
- Hugging Face model files, tokenizer cache, and large temporary outputs stay on
  the MI300X host.
- Local pulls are limited to small JSON/text summaries and logs.
- Successful run summaries are reviewed locally, then committed into
  `state.md` and `STRUCTURE.MD` when structure changes.

This keeps the workflow close to normal research engineering: code is reviewed
and versioned locally, compute happens remotely, and only compact evidence moves
back.

## Why Move Off Kaggle For The Active Path

The Kaggle path remains useful as prior work, but it is no longer the preferred
active path for Qwen3.6 validation.

Observed Kaggle constraints:

- API-pushed notebooks can land on an undesired GPU class.
- The manually selected RTX PRO 6000 notebook was internet-disabled.
- Offline bootstrapping adds machinery that does not help the core RCQ-MoE
  research loop.
- Long-running iterative jobs are awkward compared with a normal SSH host.

MI300X advantages:

- Direct SSH access.
- Persistent repository checkout and model cache.
- Normal process management for long jobs.
- Direct inspection of ROCm/PyTorch runtime state.
- Cleaner rollback through git commits and exact run directories.

Do not delete Kaggle code in this slice. Treat it as a paused backend until the
MI300X path has produced the FP smoke and one-layer RCQ evidence we need.

## Access Model

Use an SSH alias rather than placing hostnames, private keys, or credentials in
source files.

Recommended local SSH config shape:

```sshconfig
Host rcq-mi300x
  HostName <remote-host>
  User <remote-user>
  IdentityFile ~/.ssh/<dedicated-key>
  IdentitiesOnly yes
```

The alias name can be shared in the working session. Private keys, root
passwords, Hugging Face tokens, and cloud-provider credentials must not be
committed or pasted into docs.

Remote secret handling:

- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is configured only on the MI300X host.
- The repo must not print token values.
- The repo may record whether token-like environment variables are present.
- GitHub access should prefer public clone or a credential helper already
  configured on the remote.

## Remote Directory Layout

Use stable directories on the MI300X host:

```text
~/rcq/
├── repo/                  # git checkout of https://github.com/AarushCodes/rcq.git
├── hf_cache/              # HF_HOME / TRANSFORMERS_CACHE
├── outputs/
│   ├── control_plane/
│   ├── qwen36_fp_smoke/
│   └── one_layer_rcq/
└── logs/
```

Large generated files stay under `~/rcq/` on the remote. Local sync should copy
only reviewed summaries, for example:

```text
metadata.json
module_structure.txt
fp_metrics.json
run_manifest.json
stdout.log
stderr.log
```

Do not copy safetensors, PyTorch checkpoints, Hugging Face cache directories, or
large activation dumps back to the local machine.

## Environment Strategy

The DigitalOcean image is expected to provide PyTorch 2.6 plus ROCm 7.0. Do not
blindly run a dependency sync that might replace the ROCm PyTorch build with an
incompatible CPU/CUDA wheel.

Preferred setup:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e . --no-deps
python -m pip install accelerate datasets huggingface-hub numpy safetensors scipy tokenizers transformers
```

Before loading Qwen weights, record:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Even on ROCm, PyTorch exposes AMD GPUs through the `torch.cuda` API. Code should
therefore stay PyTorch-device generic and avoid NVIDIA-only assumptions in
metadata collection.

## Commit And Run Discipline

Each remote run should have a local commit boundary.

1. Make the local code change.
2. Run the relevant local tests or dry-run.
3. Update `state.md` and `STRUCTURE.MD` if the slice succeeds.
4. Commit locally.
5. Push the commit to GitHub.
6. On MI300X, fetch and check out that exact commit SHA.
7. Run one named smoke command.
8. Pull only small result files.
9. Interpret results locally.
10. Commit the result summary.

If a remote run fails, record the failure in `state.md` only when it changes the
project state or reveals a real constraint. Do not bury failed exploratory
commands in source history unless they inform the next reviewed slice.

## Initial Remote Commands

### Control Plane Smoke

Purpose: verify SSH, checkout, ROCm visibility, package imports, and dry-run
output without loading model weights.

Remote command shape:

```bash
cd ~/rcq/repo
git fetch origin main
git checkout <reviewed-commit-sha>
source .venv/bin/activate
export HF_HOME=~/rcq/hf_cache
export TRANSFORMERS_CACHE=~/rcq/hf_cache
python scripts/qwen36_fp_smoke.py \
  --dry-run \
  --output-dir ~/rcq/outputs/control_plane/<commit-sha>
```

Expected small outputs:

- `metadata.json`
- `module_structure.txt`
- `fp_metrics.json`

### FP-Only Qwen3.6 Smoke

Purpose: load the pretrained checkpoint on MI300X, inspect MoE structure, and
run a tiny next-token loss pass without quantization.

Remote command shape:

```bash
cd ~/rcq/repo
git checkout <reviewed-commit-sha>
source .venv/bin/activate
export HF_HOME=~/rcq/hf_cache
export TRANSFORMERS_CACHE=~/rcq/hf_cache
python scripts/qwen36_fp_smoke.py \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --dtype bfloat16 \
  --device-map auto \
  --sequence-length 256 \
  --eval-batch-size 1 \
  --eval-batches 1 \
  --output-dir ~/rcq/outputs/qwen36_fp_smoke/<commit-sha>
```

If the default attention implementation is unstable on ROCm, add a reviewed
script option before changing the remote command. Do not patch files manually on
the remote.

## Required Script Adaptation

Before the first real MI300X run, update the FP smoke script so metadata is not
NVIDIA-specific.

Required additions:

- record `torch.version.hip`;
- record whether ROCm tooling is present;
- attempt `rocm-smi` and `rocminfo` in addition to `nvidia-smi`;
- keep using `torch.cuda` for PyTorch device discovery;
- ensure missing `nvidia-smi` is informational, not suspicious, on ROCm.

Optional, if needed after the dry-run:

- add `--attn-implementation` passthrough to Transformers model loading;
- add a small run manifest wrapper for remote command, commit SHA, and selected
  environment variables;
- add a dedicated MI300X remote job spec format only if repeated runs become
  hard to audit with plain commands.

## Slice Plan

### Slice 1: MI300X Design

Add this design document and update project state. No remote command is run in
this slice.

### Slice 2: ROCm-Aware FP Smoke Metadata

Patch `scripts/qwen36_fp_smoke.py` to capture ROCm metadata cleanly while
remaining compatible with CUDA and local dry-runs. Run local tests.

### Slice 3: MI300X Control Plane Smoke

Use SSH to run the FP smoke script in `--dry-run` mode on MI300X from a pinned
commit. Pull only the three small output files and record the remote PyTorch,
ROCm, and GPU facts.

### Slice 4: MI300X FP-Only Qwen3.6 Smoke

Load `Qwen/Qwen3.6-35B-A3B` on MI300X, run the tiny tokenizer-driven FP pass,
and record MoE module structure plus next-token loss. Do not quantize.

### Slice 5: One-Layer Pretrained RCQ

Add or reuse layer-limited conversion. Quantize exactly one MoE layer/block on
MI300X and report FP-vs-one-layer-RCQ KL, routed MoE MSE before/after
correction, expert bpw, artifact roundtrip behavior, runtime, and peak memory.

## Failure Handling

Fail closed and preserve evidence:

- If SSH fails, do not change code for it; fix credentials outside the repo.
- If ROCm PyTorch is replaced or missing, stop and rebuild the remote virtual
  environment without altering local dependencies.
- If Qwen loading OOMs, lower sequence length or inspect device placement in a
  reviewed follow-up slice.
- If Transformers requires different loading flags on ROCm, add explicit CLI
  flags locally and commit them before retrying remotely.
- If a run produces large artifacts, leave them on MI300X and pull only compact
  summaries.

## Non-Goals

- No local copy of Qwen3.6 model weights.
- No remote code edits outside git commits.
- No credentials in source files, docs, or logs.
- No arbitrary debug shell action committed as an automation feature.
- No production kernel work before pretrained FP and one-layer RCQ evidence.
