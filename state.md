# RCQ-MoE Prototype State

Current date: 2026-05-25

## Status

The project has a working PyTorch reference prototype for RCQ-MoE, an official
Transformers Qwen3.5-MoE tiny-model integration path, and a text-derived toy
token harness with a real-tokenizer smoke path. It now also has local Kaggle
script and notebook runners for a competition-attached Qwen3.6 FP-only smoke
slice, plus a local one-shot Kaggle remote-control worker driven by reviewed
JSON job specs. The active pretrained-compatible validation plan is now moving
from Kaggle to a single AMD MI300X remote host with SSH, exact git commit
checkout, remote-only model caches, and small local result pulls. The FP smoke
script now records ROCm-aware runtime metadata needed for the first MI300X
control-plane dry-run. A Kaggle T4 x2 fallback has now produced a first
pretrained, layer-local Qwen3.6 RCQ pilot on true layer-0 MoE inputs. The
repository now has a public-facing `README.MD` copied from the current README
draft. The Down Projection V2 local implementation is now in place for both
the no-shared min-2-bit path and the left-output shared-subspace path, and the
first Kaggle T4 x2 layer-local D0/D3/D4/D5/D6 evidence run is recorded under
ignored local outputs in `runs/run2`.

This is still a research/plumbing prototype:

- Uses fake-dequant tensors, not packed/fused kernels.
- Uses tiny random official Qwen3.5-MoE models for tests.
- Has no real pretrained checkpoint quality result yet.
- Has no production storage format beyond the current reference artifact.

## Implemented

- Core RCQ math:
  - Router-weighted covariance/statistics.
  - KLT/eigendecomposition shared subspace.
  - Shared low-rank decomposition.
  - Down Projection V2 no-shared mode and shared left-output decomposition.
  - Block signed Hadamard rotations.
  - 1-bit weighted sign residual quantization.
  - 2/4-bit Lloyd-Max rescue quantization.
  - Down-specific residual configs `down_mix_1bit`, `down_min2_5p4`, and
    `down_min2_20p4`.
  - Per-expert down rotated activation moments with cold-expert shrinkage to
    global moments.
  - Effective expert bpw accounting.

- Correction:
  - Routed MoE-output affine correction.
  - Per-channel identity fallback if correction does not improve calibration MSE.

- Tiny Qwen-shaped model:
  - Packed `gate_up_proj` and separate `down_proj`.
  - Normalized top-k router.
  - Shared expert path.
  - Conversion to RCQ-backed experts.

- Official Transformers integration:
  - Uses `transformers.Qwen3_5MoeForCausalLM`.
  - Builds tiny official Qwen3.5-MoE random models.
  - Captures official MoE inputs via forward hooks.
  - Deep-copies model and replaces official sparse MoE blocks with RCQ blocks.
  - Keeps router, shared expert, attention, embeddings, norms, and LM head full precision.

- Diagnostics/harness:
  - Whole-model logits KL summary.
  - Per-layer MoE MSE before/after correction.
  - Captured shared energy.
  - 1/2/4-bit block percentages.
  - Effective bpw per expert linear.
  - Toy ablation runner.

- Artifact roundtrip:
  - Saves `metadata.json`, `non_expert_state.pt`, and `rcq_state.pt`.
  - Reloads RCQ official Qwen model without needing original FP expert tensors.
  - Converted-vs-loaded logits match exactly in toy tests.
  - Round-trips Down V2 `shared_mode` metadata and left-output tensors in the
    official-Qwen reference artifact.

- Text fixture pipeline:
  - Offline synthetic fixture source for tests.
  - Hugging Face streaming fixture builder.
  - Generated a local FineWeb-Edu 256-document fixture under ignored data.
  - Reads local text fixture files back into documents.
  - Converts text into deterministic toy byte-token `input_ids`.
  - Has a tokenizer-backed `TextTokenBatch` path with `input_ids` and
    `attention_mask`.
  - Supports optional Hugging Face tokenizer encoding in
    `scripts/official_qwen_text_smoke.py` via `--tokenizer-name-or-path`.
  - Has a checked-in tiny BERT tokenizer fixture for no-network AutoTokenizer
    unit coverage.
  - Validated slice 2 with `AutoTokenizer.from_pretrained` for
    `Qwen/Qwen3.6-35B-A3B`, then reran the smoke with
    `--tokenizer-local-files-only`.
  - Threads optional calibration/eval attention masks through the official Qwen
    capture, conversion, and KL harness paths.
  - Runs the official tiny Qwen3.5-MoE RCQ harness on text-derived tokens.
  - Saves and reloads the RCQ artifact after the text-token smoke run.

- Kaggle remote execution scaffold:
  - Adds `scripts/qwen36_fp_smoke.py` for FP-only pretrained Qwen3.6 smoke
    runs on remote GPU hardware.
  - Adds `kaggle/qwen36_fp_smoke/kernel-metadata.json` for the private Kaggle
    script kernel `aarushkhilosia/rcq-qwen36-fp-smoke`.
  - Attaches the Kaggle competition
    `nvidia-nemotron-model-reasoning-challenge` and enables GPU/internet in
    metadata.
  - Adds `kaggle/qwen36_fp_smoke/run_qwen36_fp_smoke.py`, a thin runner that
    clones the local repo from `RCQ_REPO_URL`, optionally checks out
    `RCQ_COMMIT_SHA`, installs the package, verifies `nvidia-smi`, and runs the
    FP smoke script.
  - Adds `kaggle/qwen36_fp_smoke_notebook/kernel-metadata.json` and
    `kaggle/qwen36_fp_smoke_notebook/rcq_qwen36_fp_smoke.ipynb` for a private
    notebook-kernel version of the same FP-only smoke slice.
  - The notebook can either run the committed repo smoke script through
    `RCQ_USE_REPO_RUNNER=1` plus `RCQ_REPO_URL`, or use its embedded fallback
    FP-only smoke code when no remote repo is available.
  - Kaggle API auth was configured locally in `~/.kaggle/kaggle.json`, verified
    with `kaggle config view`, and `kaggle==2.1.2` is installed in the local
    `.venv` for CLI operations.
  - Keeps credentials out of source. `RCQ_GIT_TOKEN` is optional and redacted
    from printed clone commands; `HF_TOKEN` is read only from the environment.
  - Keeps Qwen3.6 model files on Kaggle under `/kaggle/working/hf_cache`, not
    on the Mac.

- Kaggle remote-control design:
  - Adds `docs/specs/2026-05-24-kaggle-remote-control-design.md`.
  - Chooses a small JSON action framework using a public GitHub repo plus a
    separate `kaggle-jobs` branch.
  - Requires every job spec to pin the exact `main` commit through
    `code_commit`.
  - Starts with one-shot execution, not a long-running poller.
  - Initial actions are `control_plane_smoke` and `qwen36_fp_smoke`.
  - Reserves `debug_shell`, polling, GitHub result pushing, and one-layer RCQ
    conversion for later approved slices.
  - Keeps Kaggle results as output artifacts first; selected small summaries are
    committed locally after review.

- Kaggle remote-control worker:
  - Adds `rcq_moe/remote_jobs.py` for strict JSON job-spec validation,
    action/param allowlists, deterministic command construction, output-root
    containment checks, selected non-secret env snapshots, JSON writing, and
    local action execution with stdout/stderr capture.
  - Adds `kaggle/remote_worker/run_job.py`, a one-shot Kaggle entrypoint that
    clones the public repo, clones `kaggle-jobs`, reads the selected job spec,
    checks out the pinned `code_commit`, validates through the pinned code, and
    runs exactly one action.
  - Adds job templates under `kaggle/remote_worker/job_templates/` for
    `control_plane_smoke` and `qwen36_fp_smoke`.
  - Adds `tests/test_remote_jobs.py` for schema validation, reserved action
    rejection, argv construction, output containment, template validation, and a
    local dry-run action execution.
  - Still does not implement polling, `debug_shell`, GitHub result pushing, or
    one-layer RCQ conversion.

- Public GitHub remote and first job branch:
  - Renamed the GitHub repo from `kg-rcq` to `rcq`.
  - Current `origin` is `https://github.com/AarushCodes/rcq.git`.
  - Pushed local `main` to GitHub.
  - Integrated the manually added GitHub `LICENSE` commit without force-pushing.
  - Created and pushed a separate `kaggle-jobs` branch.
  - Added `jobs/0001-control-plane-smoke.json` on `kaggle-jobs`.
  - The first job runs `control_plane_smoke` and pins
    `code_commit=212fbb2816a2bf52c4da5d2bc1d7e94be3dece56`, the worker
    implementation commit.
  - No Kaggle command was run in this setup slice.

- Kaggle RTX PRO 6000 runtime constraint:
  - The manually created RTX PRO 6000 Kaggle notebook currently has internet
    disabled.
  - The existing one-shot worker path assumes internet for fetching
    `run_job.py`, cloning `main`, and cloning `kaggle-jobs`.
  - The next control-plane slice therefore needs a reviewed offline bootstrap
    path, such as a small Kaggle dataset/repo snapshot, pasted bootstrap cell,
    or Kaggle input artifact generated from a pinned GitHub commit.
  - The offline path must preserve the same guarantees: exact job JSON in
    outputs, pinned commit identity, small output artifacts, and no local copy
    of Qwen3.6 model weights.

- MI300X remote workflow design:
  - Adds `docs/specs/2026-05-25-mi300x-remote-workflow-design.md`.
  - Chooses SSH-based remote execution on a single AMD MI300X as the active
    validation path.
  - Keeps local `main` and GitHub commits as the reviewed source of truth.
  - Requires remote runs to check out an exact reviewed commit SHA.
  - Keeps Hugging Face model weights, tokenizer cache, and large temporary
    artifacts on the MI300X host.
  - Pulls only small JSON/text summaries and logs back to local.
  - Uses remote environment discipline that avoids replacing the image-provided
    PyTorch 2.6 + ROCm 7.0 build with an incompatible wheel.
  - Pauses Kaggle as a backend rather than deleting the existing Kaggle worker,
    notebook, and job-spec code.

- MI300X ROCm-aware FP smoke metadata:
  - Updates `scripts/qwen36_fp_smoke.py` dry-run metadata to include
    `torch.version.hip`, `torch.version.cuda`, PyTorch device discovery through
    `torch.cuda`, bounded `nvidia-smi`, `rocm-smi`, and `rocminfo` command
    snapshots, and a `rocm_tooling_present` flag.
  - Keeps local CPU dry-runs and CUDA hosts compatible while making missing
    `nvidia-smi` informational on ROCm-only hosts.
  - Adds test coverage through the local `control_plane_smoke` dry-run action.

- Kaggle T4 x2 single-layer Qwen3.6 RCQ pilot:
  - Adds `scripts/qwen36_single_layer_rcq_ablation.py`, a constrained
    pretrained layer-local runner for `Qwen/Qwen3.6-35B-A3B`.
  - Uses official Transformers Qwen3.5-MoE layer code on Kaggle
    `transformers==5.9.0` to run the true layer-0 token mixer:
    embeddings -> layer-0 input RMSNorm -> official layer-0 linear attention
    -> residual add -> post-attention RMSNorm -> layer-0 MoE/router.
  - Loads only tokenizer, embedding tensor, and the selected decoder layer
    shards from the Hugging Face cache; it does not load the full model.
  - Streams raw FineWeb-Edu text from Hugging Face on Kaggle, with no text
    cleanup/normalization/deduplication beyond tokenizer truncation.
  - Caches true MoE inputs, router decisions/weights, and FP routed MoE outputs
    once, then reuses them across ablations.
  - Supports partial-result resume via `--skip-completed-from`.
  - Implements Down Projection V2 D0-D7 row definitions:
    - D0 reproduces the current legacy-right `rcq_1p75` baseline.
    - D3/D4 use gate/up `rcq_1p55` with down mode `none` and min-2-bit down
      residual configs.
    - D5/D6/D7 use down `left_output` with sequential down calibration stats.
  - Sequential down calibration for D5-D7 first quantizes gate/up, then collects
    down moments and down output covariance from
    `activation(gate_Q) * up_Q` with FP down outputs.
  - Adds `kaggle_commands.md` with copy-paste Kaggle control-plane, dry-run,
    full ablation, pilot, NMSE annotation, and Down V2 time-to-insight cells.
  - Adds `tests/test_qwen36_single_layer_ablation.py` for local coverage of the
    layer-local runner helpers.

- Public README:
  - Adds `README.MD`, synced from `readme_draft.md`.
  - Summarizes RCQ-MoE motivation, research goals, compression recipe,
    related-work context, implementation status, planned validation, compute
    needs, repository map, and local reproduction commands.
  - States clearly that current tiny/random smoke results are plumbing checks,
    not pretrained-model quality evidence.
  - Does not add the local `readme_draft.md` or
    `lambda_grant_application_draft.md` drafts to git.

- Down Projection V2 local implementation:
  - Adds `SharedOutputDecomposition` and `decompose_shared_output_subspace(...)`
    for down `left_output` mode.
  - Extends storage accounting for `shared_mode` values `right_input`,
    `left_output`, and `none`.
  - Extends official-Qwen RCQ linears and artifact state to support
    `shared_mode="none"` and `shared_mode="left_output"`.
  - Extends `LinearCalibrationStats` with optional per-expert moments,
    selected-count shrinkage state, global fallback moments, router-weight
    sums, and down output covariance.
  - Adds local tests for down min-2 configs, per-expert moment handling,
    cold-expert shrinkage, left-output decomposition, official conversion,
    artifact round-trip, and layer-local D-row construction.
  - No pretrained Down V2 quality claim has been made yet; the first
    D0/D3/D4/D5/D6 Kaggle run is layer-local routed MoE-output NMSE evidence
    only.

## Generated Local Data

Generated and intentionally ignored by git:

```text
data/text_fixtures/generated/fineweb_edu_256_calib.txt  347K
data/text_fixtures/generated/fineweb_edu_256_eval.txt    80K
outputs/qwen36_fp_smoke_dry_run/
runs/run2/  # Kaggle Down V2 compact run outputs; ignored by git
```

Generation command:

```bash
python3 scripts/build_text_fixture.py \
  --source hf \
  --dataset HuggingFaceFW/fineweb-edu \
  --split train \
  --max-docs 256 \
  --max-chars-per-doc 2000 \
  --output-dir data/text_fixtures/generated \
  --prefix fineweb_edu_256
```

Generation summary:

```text
docs_total=256 docs_calib=205 docs_eval=51
```

## Validation

Latest Qwen3.6 FP smoke dry-run command:

```bash
uv run python scripts/qwen36_fp_smoke.py \
  --dry-run \
  --output-dir outputs/qwen36_fp_smoke_dry_run
```

Latest Qwen3.6 FP smoke dry-run result:

```text
metadata.json, module_structure.txt, and fp_metrics.json written successfully.
metadata.json now includes top-level cuda/nvidia_smi/rocm_smi/rocminfo fields
and a nested runtime block with torch_hip_version, torch_cuda_version,
rocm_tooling_present, bounded rocm-smi, bounded rocminfo, and bounded
nvidia-smi probe snapshots.
No pretrained model weights were loaded locally.
No Kaggle command was run.
```

Latest Kaggle notebook push command:

```bash
.venv/bin/kaggle kernels push -p kaggle/qwen36_fp_smoke_notebook
```

Latest Kaggle notebook push result:

```text
Kernel version 1 successfully pushed.
URL: https://www.kaggle.com/code/aarushkhilosia/rcq-qwen3-6-fp-smoke-notebook
```

Latest Kaggle notebook status check:

```bash
.venv/bin/kaggle kernels status aarushkhilosia/rcq-qwen3-6-fp-smoke-notebook
```

Latest Kaggle notebook status result:

```text
KernelWorkerStatus.RUNNING
```

Important interpretation:

```text
The API-pushed notebook was observed running on a P100, not RTX PRO 6000.
No useful Qwen3.6 FP output has been pulled yet.
For the RTX PRO 6000 slice, prefer a manually created Kaggle notebook with the
competition attached and the accelerator selected in the Kaggle UI.
```

Latest full test command:

```bash
uv run pytest -q
```

Latest result:

```text
81 passed
```

Latest Down Projection V2 local implementation slice:

```text
Implemented down mode none and left_output locally, including D0-D7 row
construction for the layer-local Qwen3.6 runner and Section 9 Kaggle commands
for a D0/D3/D4/D5/D6 time-to-insight run. Full local tests pass.
```

Latest GitHub push for Kaggle Down V2:

```text
origin/main=dc4f3b097369b22284bc48f2b56a29d2e6b2dbe2
Pinned Down V2 implementation commit for Kaggle:
65d1ebf867a38257412d1a6b8efd7a9856d70014
```

Latest Kaggle T4 x2 Down Projection V2 pilot:

```text
Output location: runs/run2.
Command path: kaggle_commands.md Section 9.
Runner status: ok.
Model: Qwen/Qwen3.6-35B-A3B.
Layer: 0.
Activation source: true_layer0_post_attention_norm.
Data policy: same 32 calibration docs / 8 held-out docs / 1024 max tokens per
doc policy as the prior layer-local pilot.
Held-out FP reference variance shown in pasted summary:
  0.007117566059033092.

D0_current_baseline_legacy_right_rcq_1p75_correction:
  average_expert_bpw=1.7575926781
  gate_bpw=1.7582030296, up_bpw=1.7582030296, down_bpw=1.7563719749
  heldout_mse=0.0031083712
  heldout_nmse_mean_square=0.4367076224
  down_mode=right_input, down_cfg=rcq_1p75
  down_legacy_right_captured_energy=0.0409812603
  down_widths={1: 0.75, 2: 0.2000000477, 4: 0.0499999523}

D3_gate_up_1p55_down_none_min2_5p4_correction:
  average_expert_bpw=1.8392186165
  gate_bpw=1.5682029724, up_bpw=1.5682029724, down_bpw=2.3812499046
  heldout_mse=0.0027873133
  heldout_nmse_mean_square=0.3916009069
  down_mode=none, down_cfg=down_min2_5p4
  down_widths={1: 0.0, 2: 0.9500000477, 4: 0.0499999523}

D4_gate_up_1p55_down_none_min2_20p4_correction:
  average_expert_bpw=1.9392186801
  gate_bpw=1.5682029724, up_bpw=1.5682029724, down_bpw=2.6812500954
  heldout_mse=0.0026753739
  heldout_nmse_mean_square=0.3758740844
  down_mode=none, down_cfg=down_min2_20p4
  down_widths={1: 0.0, 2: 0.7999999523, 4: 0.2000000477}

D5_gate_up_1p55_down_left_output_mix_1bit:
  average_expert_bpw=1.5997980436
  gate_bpw=1.5682029724, up_bpw=1.5682029724, down_bpw=1.6629881859
  heldout_mse=0.0029326434
  heldout_nmse_mean_square=0.4120189131
  down_mode=left_output, down_cfg=down_mix_1bit
  down_left_output_captured_energy=0.0680184296
  down_widths={1: 0.75, 2: 0.2000000477, 4: 0.0499999523}

D6_gate_up_1p55_down_left_output_min2_5p4:
  average_expert_bpw=1.8497980436
  gate_bpw=1.5682029724, up_bpw=1.5682029724, down_bpw=2.4129881859
  heldout_mse=0.0027842524
  heldout_nmse_mean_square=0.3911708661
  down_mode=left_output, down_cfg=down_min2_5p4
  down_left_output_captured_energy=0.0680184296
  down_widths={1: 0.0, 2: 0.9500000477, 4: 0.0499999523}

D7_gate_up_1p75_down_left_output_min2_20p4:
  status=not_available
  reason=intentionally skipped by Section 9 time-to-insight Down V2 cell.
```

Interpretation of the pasted Down V2 summary:

```text
D6 is the best recorded <=1.90 average-bpw row in the pasted summary:
  heldout_nmse=0.3911708661 at average_expert_bpw=1.8497980436.
  Relative improvement vs D0 is about 10.4%.

D3 is the immediate no-shared min-2-bit down fix at <=1.90 average bpw:
  heldout_nmse=0.3916009069 at average_expert_bpw=1.8392186165.
  D6 improves over D3 by only about 0.11% relative NMSE, so this pilot does
  not justify paying left-output shared overhead over `none + down_min2_5p4`.

D4 has the lowest pasted NMSE:
  heldout_nmse=0.3758740844,
  but average_expert_bpw=1.9392186801 exceeds the <=1.90 target.

D5 is lower bpw than D0 but only modestly improves NMSE:
  heldout_nmse=0.4120189131 at average_expert_bpw=1.5997980436.

Neither recorded <=1.90-bpw row reaches the Down V2 minimum success criterion
of heldout NMSE <=0.33. The new left-output down subspace captures about
6.8% routed down-output energy, better than the old 4.1% legacy-right
captured-energy diagnostic, but still small.

These are layer-local routed MoE-output NMSE results only, not full-model KL,
PPL, or downstream quality.
```

Latest remote-worker focused test command:

```bash
uv run pytest tests/test_remote_jobs.py -q
```

Latest remote-worker focused test result:

```text
20 passed
```

Latest remote-worker implementation slice:

```text
Added local job validation/dispatch, one-shot Kaggle runner, job templates, and
tests. No Kaggle command was run and no pretrained model weights were loaded.
```

Latest GitHub remote setup:

```text
origin=https://github.com/AarushCodes/rcq.git
main pushed to GitHub.
kaggle-jobs pushed with jobs/0001-control-plane-smoke.json pinned to
212fbb2816a2bf52c4da5d2bc1d7e94be3dece56.
```

Latest Kaggle architecture update:

```text
Manual RTX PRO 6000 Kaggle notebook has internet disabled, so the GitHub-clone
worker path cannot run there as-is. Next slice should add an offline bootstrap
path before running control_plane_smoke.
```

Latest README slice:

```text
README.MD resynced from readme_draft.md. The tiny-model KL wording was tightened
so the README no longer presents the tiny/random KL value as a quality result.
No tests were run because this was a documentation-only slice.
```

Latest MI300X design slice:

```text
Added docs/specs/2026-05-25-mi300x-remote-workflow-design.md and updated
state/structure to make MI300X the active remote validation path. No SSH command
was run, no remote access was used, and no pretrained model weights were loaded.
No tests were run because this was a documentation-only slice.
```

Latest MI300X metadata slice:

```text
Patched scripts/qwen36_fp_smoke.py to record ROCm-aware runtime metadata for
MI300X dry-runs while preserving CUDA/local CPU compatibility. The local dry-run
completed without loading pretrained model weights. No SSH command was run and
no remote access was used.
```

Latest Kaggle T4 x2 single-layer Qwen3.6 RCQ pilot:

```text
Environment: Kaggle 2 x Tesla T4, Python 3.12.12, torch 2.10.0+cu128,
transformers 5.9.0.
Source commit used by runner: latest main through the Kaggle command cell.
Model: Qwen/Qwen3.6-35B-A3B.
Layer: 0, layer_type=linear_attention.
Activation source: true_layer0_post_attention_norm.
Path: embed_tokens -> layer0 input RMSNorm -> layer0 attention -> residual add
-> post-attention RMSNorm.
Data: raw HuggingFaceFW/fineweb-edu train stream.
Calibration: first 32 streamed docs.
Held-out eval: next 8 streamed docs.
Max tokens per doc: 1024.
Text postprocessing: none; raw dataset text is passed directly to tokenizer
with truncation only.
Elapsed: 1987.6756 sec.
```

Latest Kaggle T4 x2 NMSE denominator recovery:

```text
Command path: kaggle_commands.md Section 8, using
scripts/qwen36_single_layer_rcq_ablation.py --denominator-only with the
existing pilot ablation_metrics.json.
Purpose: recompute only FP reference routed MoE-output denominators from the
same layer-local activation path and streamed document policy, then annotate
the existing metrics JSON with NMSE. It did not rebuild quantized ablations.
Runner exit status: 0.
Calibration FP reference:
  count=43266048, mean=-0.0003666695, mean_square=0.0082688514,
  variance=0.0082687169, rms=0.0909332248, max_abs=4.06640625.
Held-out FP reference:
  count=8517632, mean=-0.0004167252, mean_square=0.0071177397,
  variance=0.0071175661, rms=0.0843666979, max_abs=3.466796875.
```

Latest Kaggle T4 x2 pilot held-out routed MoE-output MSE/RMSE:

```text
baseline_fp_layer_local: mse=0, rmse=0
A0 shared + naive 1-bit residual, no Hadamard/rescue/correction:
  mse=0.0050930439, rmse=0.0713655649
A1 A0 + Hadamard:
  mse=0.0046207301, rmse=0.0679759525
A2 A1 + activation-weighted binary scale:
  mse=0.0046217143, rmse=0.0679831910
A3 A2 + rcq_1p55 rescue:
  mse=0.0037782405, rmse=0.0614673935
A3 A2 + rcq_1p75 rescue:
  mse=0.0031429059, rmse=0.0560616255
A3 A2 + rcq_1p90 rescue:
  mse=0.0026224348, rmse=0.0512097136
A4 rcq_1p55 + routed MoE-output correction:
  mse=0.0037350447, rmse=0.0611150119
A4 rcq_1p75 + routed MoE-output correction:
  mse=0.0031083712, rmse=0.0557527684
A4 rcq_1p90 + routed MoE-output correction:
  mse=0.0025962735, rmse=0.0509536408
```

Latest Kaggle T4 x2 pilot held-out routed MoE-output NMSE:

```text
Denominator: held-out FP routed MoE-output mean square = 0.0071177397.
Variance denominator gives nearly identical values because the held-out FP
output mean is close to zero.

baseline_fp_layer_local: nmse_mean_square=0, nmse_variance=0
A0 shared + naive 1-bit residual, no Hadamard/rescue/correction:
  nmse_mean_square=0.7155423012, nmse_variance=0.7155597595
A1 A0 + Hadamard:
  nmse_mean_square=0.6491850362, nmse_variance=0.6492008756
A2 A1 + activation-weighted binary scale:
  nmse_mean_square=0.6493233025, nmse_variance=0.6493391452
A3 A2 + rcq_1p55 rescue:
  nmse_mean_square=0.5308202620, nmse_variance=0.5308332133
A3 A2 + rcq_1p75 rescue:
  nmse_mean_square=0.4415595369, nmse_variance=0.4415703105
A3 A2 + rcq_1p90 rescue:
  nmse_mean_square=0.3684364524, nmse_variance=0.3684454418
A4 rcq_1p55 + routed MoE-output correction:
  nmse_mean_square=0.5247515120, nmse_variance=0.5247643153
A4 rcq_1p75 + routed MoE-output correction:
  nmse_mean_square=0.4367076224, nmse_variance=0.4367182776
A4 rcq_1p90 + routed MoE-output correction:
  nmse_mean_square=0.3647609506, nmse_variance=0.3647698503
```

Interpretation:

```text
This is the first pretrained Qwen3.6 RCQ evidence, but it is layer-local only.
It is not full-model KL, PPL, or downstream quality. It shows the expected
directional pattern on true layer-0 MoE inputs: Hadamard improves over naive
1-bit; mixed-bit rescue gives the largest improvement; routed correction gives
a small additional held-out improvement for each rescue setting. The held-out
NMSE values are still high in absolute terms for layer-local output
replacement, so these results are useful for ablation direction and debugging,
not yet a quality claim.
```

Latest text-token smoke command:

```bash
uv run python scripts/official_qwen_text_smoke.py \
  --clean \
  --tokenizer-name-or-path Qwen/Qwen3.6-35B-A3B \
  --tokenizer-local-files-only \
  --output-dir outputs/toy_official_qwen_text_smoke_qwen36_tokenizer
```

Latest text-token smoke result:

```text
text_source=files:/Users/ck/Desktop/aarush/inference_research/impl/data/text_fixtures/generated/fineweb_edu_256_calib.txt,/Users/ck/Desktop/aarush/inference_research/impl/data/text_fixtures/generated/fineweb_edu_256_eval.txt
tokenization=hf_tokenizer:Qwen/Qwen3.6-35B-A3B
calib_docs=205 eval_docs=51
calibration_ids_shape=(16, 32) eval_ids_shape=(4, 32)
calibration_attention_tokens=512 eval_attention_tokens=128
toy_text_fp_vs_rcq_kl=mean=6.1713831e-07, p50=4.7719027e-07, p95=4.4415056e-06, p99=5.9478944e-06, max=9.4008465e-06
layer=0 moe_mse_before=9.1268459e-09 moe_mse_after=8.7941094e-09 bpw={'gate': 8.59375, 'up': 8.59375, 'down': 8.59375}
layer=1 moe_mse_before=1.0904087e-08 moe_mse_after=1.085529e-08 bpw={'gate': 8.59375, 'up': 8.59375, 'down': 8.59375}
artifact_dir=outputs/toy_official_qwen_text_smoke_qwen36_tokenizer
max_abs_logit_delta_converted_vs_loaded=0
kl_converted_vs_loaded=mean=0, p50=0, p95=0, p99=0, max=0
```

Interpretation: this is a tiny random-model plumbing metric only. It verifies
text loading, Qwen tokenizer loading/cached local reuse, attention-mask plumbing,
official Qwen RCQ conversion, diagnostics, artifact save, and artifact reload.
It is not evidence of pretrained model quality.

Known warnings:

- Two SWIG deprecation warnings from optional dependencies during official
  Transformers model import. These do not currently affect tests.

## Environment

Important package versions:

```text
torch 2.12.0
torchvision 0.27.0
transformers 5.9.0
accelerate 1.13.0
safetensors 0.7.0
datasets 4.8.5
tokenizers 0.22.2
huggingface_hub 1.16.1
scikit-learn 1.8.0
numba 0.65.1
librosa 0.11.0
kaggle 2.1.2  # installed in local .venv for CLI auth/push/status
```

Important constraint:

```text
transformers 5.9.0 requires tokenizers >=0.22.0, <=0.23.0
```

Do not blindly upgrade `tokenizers` beyond `0.23.0` unless `transformers` is
also upgraded to a compatible version.

`uv.lock` is tracked for environment reproducibility.

## Useful Scripts

```bash
python3 scripts/official_qwen_toy_kl.py
python3 scripts/official_qwen_toy_ablation.py
python3 scripts/official_qwen_toy_artifact.py --clean
python3 scripts/official_qwen_text_smoke.py --clean
python3 scripts/qwen36_fp_smoke.py --dry-run --output-dir outputs/qwen36_fp_smoke_dry_run
python3 scripts/build_text_fixture.py --source synthetic --prefix synthetic_toy --output-dir outputs/text_fixture_smoke
```

HF streaming fixture:

```bash
python3 scripts/build_text_fixture.py \
  --source hf \
  --dataset HuggingFaceFW/fineweb-edu \
  --split train \
  --max-docs 256 \
  --max-chars-per-doc 2000 \
  --output-dir data/text_fixtures/generated \
  --prefix fineweb_edu_256
```

## Recent Commits Before This State Update

```text
dc4f3b0 Add Kaggle Down V2 pilot commands
65d1ebf Add down left-output projection path
bea155b Add down projection min2 no-shared path
f4490a2 Record Kaggle pilot NMSE results
e29b008 Add NMSE reporting for Qwen layer ablations
1a4d5ea Record Kaggle T4 Qwen layer pilot
f68645f Resume completed ablation rows
fcc479d Cache single-layer ablation inputs
24b5be0 Add verbose Kaggle ablation progress logs
1216480 Use official Qwen layer for single-layer calibration
02338c0 Avoid AutoConfig for Qwen single-layer inspector
d951986 Add Kaggle single-layer Qwen RCQ ablation runner
70dd76c Add ROCm-aware Qwen FP smoke metadata
fa9d25e Record GitHub repo rename
81a0599 Resync public README
790d488 Add public README
039b6c5 Record Kaggle offline bootstrap constraint
4560a4f Record GitHub remote setup
42a158d Create LICENSE
212fbb2 Add one-shot Kaggle remote worker
8490012 Design Kaggle remote control workflow
d370d5a Add Kaggle Qwen3.6 FP smoke notebook
de335b9 Add Kaggle Qwen3.6 FP smoke runner
b9a3a8e Add Qwen tokenizer smoke coverage
2183cca Update RCQ prototype state
3f3b713 Add tokenizer-backed text batches
54eb9cb Document current RCQ prototype state
e872141 Add streamed text fixture builder
```

## Current Limitations

- No full pretrained Qwen/MoE checkpoint has been quantized.
- There is now a pretrained Qwen3.6 layer-local RCQ pilot for layer 0 on
  Kaggle T4 x2, but no full-model pretrained quality evaluation yet.
- Down Projection V2 has a first Kaggle T4 x2 layer-local run in `runs/run2`,
  but D3 and D6 stay around held-out NMSE 0.391 at <=1.90 average bpw, above
  the <=0.33 success threshold.
- The API-pushed competition-attached Kaggle notebook selected P100 rather than
  RTX PRO 6000, so it is not the desired validation path for Slice 3.
- The manually created RTX PRO 6000 Kaggle notebook currently has internet
  disabled, so the GitHub-clone worker path cannot run there as-is.
- The MI300X path is designed and the FP smoke script is ROCm-metadata-ready,
  but it has not yet been executed over SSH; no SSH alias, remote PyTorch/ROCm
  metadata, or remote dry-run output has been recorded yet.
- No completed full-model Kaggle Qwen3.6 FP smoke outputs have been pulled or
  interpreted.
- Current artifact stores fake-dequant reference tensors, not packed bitstreams.
- No FP8 scale/shared-factor storage.
- No fused kernels or performance benchmarks.
- No downstream task evaluation.
- No real PPL evaluation.
- No grouped subspaces ablation.

## Recommended Next Milestone

Continue the pretrained-compatible validation path slice by slice:

1. Extend the Kaggle T4 x2 layer-local pilot.
   - D3 is now recovered from the compact pasted summary; verify
     `runs/run2/ablation_metrics.json` only if exact JSON provenance is needed.
   - D6 improves over D3 by only about 0.11% relative NMSE, so do not spend
     more effort on left-output shared overhead unless a larger run contradicts
     this pilot.
   - Run D7 only as a high-bpw floor check if runtime is cheap; otherwise move
     to a neuron-alignment research slice.
   - Add compact doc hashes/lengths for the streamed 40-document pilot set.
   - Consider a larger 256/64/4096 evidence run if runtime remains acceptable.
   - Still report this as layer-local routed MoE-output MSE, not full-model KL.
2. MI300X control-plane smoke when MI300X capacity is available.
   - Use SSH to run `scripts/qwen36_fp_smoke.py --dry-run` from an exact pinned
     commit on the MI300X host.
   - Verify PyTorch, ROCm, GPU visibility, repo checkout, imports, and small
     output writing.
   - Do not load Qwen weights in this slice.
3. MI300X Qwen3.6 FP-only smoke.
   - Load `Qwen/Qwen3.6-35B-A3B` on MI300X, not locally.
   - Use the remote Hugging Face cache under the MI300X workspace.
   - Run tokenizer-driven FP-only eval and inspect/capture sparse MoE block
     structure under the remote output directory.
   - Pull only compact JSON/text/log outputs back to local.
   - Do not quantize in this slice.
4. One-layer pretrained RCQ conversion in a full-model context.
   - Add layer-limited conversion if needed.
   - Quantize exactly one Qwen3.6 MoE layer/block first on MI300X.
   - Report FP-vs-one-layer-RCQ KL, routed MoE MSE before/after correction,
     expert bpw, artifact save/load exactness, and memory/runtime notes.
