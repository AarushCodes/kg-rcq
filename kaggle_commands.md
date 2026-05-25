# Kaggle Command Cells

Copy these cells into a Kaggle notebook. Keep large model files and caches on
Kaggle. Pull back only compact logs, JSON, and text outputs.

## 1. Control-Plane Smoke

Purpose: verify Kaggle GPU visibility, clone the reviewed repo commit, check
the relevant Python package versions, and run the Qwen3.6 dry-run metadata
script. This does not install the local package and does not load pretrained
model weights.

```bash
%%bash
set -euo pipefail

export RCQ_COMMIT=70dd76c
export RCQ_ROOT=/kaggle/working/rcq
export RCQ_LOG_DIR=/kaggle/working/rcq_logs
mkdir -p "$RCQ_LOG_DIR"

exec > >(stdbuf -oL tee -a "$RCQ_LOG_DIR/bootstrap.log") 2>&1

echo "=== date ==="
date -u

echo "=== gpu ==="
nvidia-smi || true

echo "=== python ==="
python --version
python - <<'PY'
import platform
import torch

print("platform", platform.platform())
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("device", i, torch.cuda.get_device_name(i))
PY

echo "=== clone repo ==="
rm -rf "$RCQ_ROOT"
git clone https://github.com/AarushCodes/rcq.git "$RCQ_ROOT"
cd "$RCQ_ROOT"
git checkout "$RCQ_COMMIT"
git rev-parse HEAD

echo "=== dependency versions ==="
python - <<'PY'
for name in ["accelerate", "datasets", "huggingface_hub", "safetensors", "tokenizers", "transformers"]:
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(name, "IMPORT_FAILED", repr(exc))
PY

echo "=== dry run ==="
PYTHONPATH="$RCQ_ROOT" python scripts/qwen36_fp_smoke.py \
  --dry-run \
  --output-dir /kaggle/working/outputs/kaggle_t4_control_plane

echo "=== output files ==="
find /kaggle/working/outputs/kaggle_t4_control_plane -maxdepth 1 -type f -print -exec wc -c {} \;

echo "=== compact metadata summary ==="
python - <<'PY'
import json

p = "/kaggle/working/outputs/kaggle_t4_control_plane/metadata.json"
d = json.load(open(p))
r = d["runtime"]
print("torch", d["torch"])
print("cuda_available", r["cuda"]["cuda_available"])
print("device_count", r["cuda"]["device_count"])
print("devices", [x["name"] for x in r["cuda"].get("devices", [])])
print("nvidia_smi_available", r["nvidia_smi"]["available"])
print("rocm_tooling_present", r["rocm_tooling_present"])
PY

echo "=== package results ==="
cd /kaggle/working
tar -czf rcq_control_plane_outputs.tgz outputs/kaggle_t4_control_plane rcq_logs/bootstrap.log
ls -lh rcq_control_plane_outputs.tgz
```

Paste back:

- the `=== compact metadata summary ===` section;
- any error trace if the cell fails;
- whether `nvidia-smi` reports `2 x T4`.

## 2. Download Result Bundle In Kaggle UI

After the first cell succeeds, download:

```text
/kaggle/working/rcq_control_plane_outputs.tgz
```

Do not download model weights, Hugging Face cache directories, safetensors, or
large activation dumps.

## 3. Single-Layer Qwen3.6 RCQ Ablation Dry-Run

Purpose: inspect Qwen3.6 config plus safetensors index for the selected layer
without loading tensor shards or text data. This should run before the real
ablation cell.

```bash
%%bash
set -euo pipefail

export RCQ_REPO_URL=https://github.com/AarushCodes/rcq.git
export RCQ_COMMIT="$(git ls-remote "$RCQ_REPO_URL" refs/heads/main | awk '{print $1}')"
export RCQ_ROOT=/kaggle/working/rcq
export HF_HOME=/kaggle/working/hf_cache
export TRANSFORMERS_CACHE=/kaggle/working/hf_cache
export RCQ_OUT=/kaggle/working/outputs/qwen36_single_layer_rcq_dry_run
mkdir -p /kaggle/working/outputs "$HF_HOME"

echo "=== selected commit ==="
echo "$RCQ_COMMIT"

echo "=== clone repo ==="
rm -rf "$RCQ_ROOT"
git clone "$RCQ_REPO_URL" "$RCQ_ROOT"
cd "$RCQ_ROOT"
git checkout "$RCQ_COMMIT"
git rev-parse HEAD

echo "=== ensure qwen3.5 moe transformers support ==="
python -m pip install -U "transformers>=5.9.0,<6"
python - <<'PY'
import transformers
print("transformers", transformers.__version__)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
print("qwen3_5_moe_decoder_layer_import", Qwen3_5MoeDecoderLayer.__name__)
PY

echo "=== dry-run inspect ==="
PYTHONUNBUFFERED=1 PYTHONPATH="$RCQ_ROOT" stdbuf -oL -eL python -u scripts/qwen36_single_layer_rcq_ablation.py \
  --dry-run \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --layer-id 0 \
  --verbose \
  --output-dir "$RCQ_OUT" \
  --cache-dir "$HF_HOME"

echo "=== compact index summary ==="
python - <<'PY'
import json
p = "/kaggle/working/outputs/qwen36_single_layer_rcq_dry_run/index_summary.json"
d = json.load(open(p))
print("model_type", d.get("model_type"))
print("hidden_size", d.get("hidden_size"))
print("moe_intermediate_size", d.get("moe_intermediate_size"))
print("num_hidden_layers", d.get("num_hidden_layers"))
print("num_experts", d.get("num_experts"))
print("num_experts_per_tok", d.get("num_experts_per_tok"))
print("layer_type", d.get("layer_type"))
print("attention_bias", d.get("attention_bias"))
print("rope_parameters", d.get("rope_parameters"))
print("missing_required_tensors", d.get("missing_required_tensors"))
print("expected_layer_tensor_count", d.get("expected_layer_tensor_count"))
print("matched_layer_tensor_count", d.get("matched_layer_tensor_count"))
print("matched_embedding_key", d.get("matched_embedding_key"))
print("layer_type_schedule_prefix", d.get("layer_type_schedule_prefix"))
print("required_shard_count", d["required_tensor_summary"]["required_shard_count"])
print("required_shards", d["required_tensor_summary"]["required_shards"])
PY
```

Paste back the `=== compact index summary ===` section before running the real
ablation cell. If `layer_type` is not `full_attention` or required tensors are
missing, stop.

## 4. Single-Layer Qwen3.6 RCQ Ablation

Purpose: run layer-local pretrained RCQ ablations on true layer-0 MoE inputs:

```text
tokens -> embeddings -> layer0 input RMSNorm -> layer0 attention
       -> residual add -> layer0 post-attention RMSNorm -> layer0 MoE/router
```

It streams raw FineWeb-Edu text, uses first 256 docs for calibration and next
64 docs for held-out evaluation, truncates each document to 4096 tokenizer
tokens, and does no text cleanup/normalization/deduplication.

```bash
%%bash
set -euo pipefail

export RCQ_REPO_URL=https://github.com/AarushCodes/rcq.git
export RCQ_COMMIT="$(git ls-remote "$RCQ_REPO_URL" refs/heads/main | awk '{print $1}')"
export RCQ_ROOT=/kaggle/working/rcq
export HF_HOME=/kaggle/working/hf_cache
export TRANSFORMERS_CACHE=/kaggle/working/hf_cache
export RCQ_OUT=/kaggle/working/outputs/qwen36_single_layer_rcq_ablation
export RCQ_LOG_DIR=/kaggle/working/rcq_logs
mkdir -p /kaggle/working/outputs "$HF_HOME" "$RCQ_LOG_DIR"

exec > >(stdbuf -oL tee -a "$RCQ_LOG_DIR/qwen36_single_layer_rcq_ablation.log") 2>&1

echo "=== selected commit ==="
echo "$RCQ_COMMIT"

echo "=== gpu ==="
nvidia-smi || true

echo "=== clone repo ==="
rm -rf "$RCQ_ROOT"
git clone "$RCQ_REPO_URL" "$RCQ_ROOT"
cd "$RCQ_ROOT"
git checkout "$RCQ_COMMIT"
git rev-parse HEAD

echo "=== ensure qwen3.5 moe transformers support ==="
python -m pip install -U "transformers>=5.9.0,<6"
python - <<'PY'
import transformers
print("transformers", transformers.__version__)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
print("qwen3_5_moe_decoder_layer_import", Qwen3_5MoeDecoderLayer.__name__)
PY

echo "=== run ablations ==="
PYTHONUNBUFFERED=1 PYTHONPATH="$RCQ_ROOT" stdbuf -oL -eL python -u scripts/qwen36_single_layer_rcq_ablation.py \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --layer-id 0 \
  --calib-docs 256 \
  --eval-docs 64 \
  --max-tokens-per-doc 4096 \
  --activation-source true_layer0_post_attention_norm \
  --verbose \
  --output-dir "$RCQ_OUT" \
  --cache-dir "$HF_HOME"

echo "=== compact ablation summary ==="
python - <<'PY'
import json
p = "/kaggle/working/outputs/qwen36_single_layer_rcq_ablation/ablation_metrics.json"
d = json.load(open(p))
print("status", d["status"])
print("activation_source", d["activation_source"])
print("elapsed_sec", d["elapsed_sec"])
for row in d["results"]:
    if row["status"] != "ok":
        print(row["label"], row["status"], row.get("reason"))
        continue
    held = row.get("heldout", {})
    print(row["label"], "heldout_mse", held.get("mse"), "heldout_rmse", held.get("rmse"))
PY

echo "=== package compact results ==="
cd /kaggle/working
tar -czf qwen36_single_layer_rcq_ablation_outputs.tgz \
  outputs/qwen36_single_layer_rcq_ablation/run_manifest.json \
  outputs/qwen36_single_layer_rcq_ablation/index_summary.json \
  outputs/qwen36_single_layer_rcq_ablation/ablation_metrics.json \
  rcq_logs/qwen36_single_layer_rcq_ablation.log
ls -lh qwen36_single_layer_rcq_ablation_outputs.tgz
```

Download only:

```text
/kaggle/working/qwen36_single_layer_rcq_ablation_outputs.tgz
```

Do not download model weights, Hugging Face cache directories, safetensors, or
large activation dumps.

## 5. Experiment Interpretation

This experiment is layer-local pretrained evidence, not full-model quality.

The honest claim is:

```text
For Qwen3.6 layer 0, using true layer-0 post-attention MoE inputs from raw
FineWeb-Edu text, RCQ variants produce these routed MoE-output errors and bpw
diagnostics.
```

Do not claim full-model KL, PPL, or downstream quality from this slice.

## 6. Best Time-To-Insight Pilot

Purpose: run a proper, non-proxy pilot before the full 256/64/4096 ablation.
This uses true layer-0 MoE inputs through the official layer-0 token mixer,
streams real FineWeb-Edu text, keeps a held-out split, and runs the most useful
early RCQ ablations.

Settings:

```text
calibration: first 32 streamed FineWeb-Edu docs
held-out eval: next 8 streamed FineWeb-Edu docs
max tokens per doc: 1024
activation source: true_layer0_post_attention_norm
experiments: first 13 rows, through A4 rescue configs
```

```bash
%%bash
set -euo pipefail

export RCQ_REPO_URL=https://github.com/AarushCodes/rcq.git
export RCQ_COMMIT="$(git ls-remote "$RCQ_REPO_URL" refs/heads/main | awk '{print $1}')"
export RCQ_ROOT=/kaggle/working/rcq
export HF_HOME=/kaggle/working/hf_cache
export TRANSFORMERS_CACHE=/kaggle/working/hf_cache
export RCQ_OUT=/kaggle/working/outputs/qwen36_single_layer_rcq_pilot
export RCQ_LOG_DIR=/kaggle/working/rcq_logs
mkdir -p /kaggle/working/outputs "$HF_HOME" "$RCQ_LOG_DIR"

exec > >(stdbuf -oL tee -a "$RCQ_LOG_DIR/qwen36_single_layer_rcq_pilot.log") 2>&1

echo "=== selected commit ==="
echo "$RCQ_COMMIT"

echo "=== gpu ==="
nvidia-smi || true

echo "=== clone repo ==="
rm -rf "$RCQ_ROOT"
git clone "$RCQ_REPO_URL" "$RCQ_ROOT"
cd "$RCQ_ROOT"
git checkout "$RCQ_COMMIT"
git rev-parse HEAD

echo "=== ensure qwen3.5 moe transformers support ==="
python -m pip install -U "transformers>=5.9.0,<6"
python - <<'PY'
import transformers
print("transformers", transformers.__version__)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
print("qwen3_5_moe_decoder_layer_import", Qwen3_5MoeDecoderLayer.__name__)
PY

echo "=== run pilot ablations ==="
PYTHONUNBUFFERED=1 PYTHONPATH="$RCQ_ROOT" python -u scripts/qwen36_single_layer_rcq_ablation.py \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --layer-id 0 \
  --calib-docs 32 \
  --eval-docs 8 \
  --max-tokens-per-doc 1024 \
  --activation-source true_layer0_post_attention_norm \
  --max-experiments 13 \
  --verbose \
  --output-dir "$RCQ_OUT" \
  --cache-dir "$HF_HOME" \
  > "$RCQ_LOG_DIR/qwen36_single_layer_rcq_pilot.runner.log" 2>&1 &

RUN_PID=$!
echo "runner_pid=$RUN_PID"

echo "=== streaming runner log ==="
while kill -0 "$RUN_PID" 2>/dev/null; do
  tail -n 80 "$RCQ_LOG_DIR/qwen36_single_layer_rcq_pilot.runner.log" || true
  echo "=== still running $(date -u) ==="
  sleep 20
done

wait "$RUN_PID"
RUN_STATUS=$?
echo "runner_exit_status=$RUN_STATUS"
echo "=== final runner log ==="
tail -n 240 "$RCQ_LOG_DIR/qwen36_single_layer_rcq_pilot.runner.log" || true
if [ "$RUN_STATUS" -ne 0 ]; then
  exit "$RUN_STATUS"
fi

echo "=== compact pilot summary ==="
python - <<'PY'
import json

p = "/kaggle/working/outputs/qwen36_single_layer_rcq_pilot/ablation_metrics.json"
d = json.load(open(p))
print("status", d["status"])
print("activation_source", d["activation_source"])
print("activation_source_detail", d["activation_source_detail"])
print("doc_policy", d["doc_policy"])
print("elapsed_sec", d["elapsed_sec"])
for row in d["results"]:
    if row["status"] != "ok":
        print(row["label"], row["status"], row.get("reason"))
        continue
    held = row.get("heldout", {})
    print(row["label"], "heldout_mse", held.get("mse"), "heldout_rmse", held.get("rmse"))
PY

echo "=== package compact results ==="
cd /kaggle/working
tar -czf qwen36_single_layer_rcq_pilot_outputs.tgz \
  outputs/qwen36_single_layer_rcq_pilot/run_manifest.json \
  outputs/qwen36_single_layer_rcq_pilot/index_summary.json \
  outputs/qwen36_single_layer_rcq_pilot/ablation_metrics.json \
  rcq_logs/qwen36_single_layer_rcq_pilot.log \
  rcq_logs/qwen36_single_layer_rcq_pilot.runner.log
ls -lh qwen36_single_layer_rcq_pilot_outputs.tgz
```

Paste back:

- the `=== compact pilot summary ===` section;
- any error trace if the cell fails;
- whether the result bundle was created.

## 7. Best Time-To-Insight Pilot As Python Cell

Use this if Kaggle buffers all `%%bash` output and you do not even see early
`echo` lines. Paste this into a normal Kaggle Python cell, not a `%%bash` cell.

```python
import os
import subprocess
import time
from pathlib import Path


def run(cmd, cwd=None, check=True):
    print(f"\n=== $ {cmd} ===", flush=True)
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")
    return result


RCQ_REPO_URL = "https://github.com/AarushCodes/rcq.git"
RCQ_ROOT = Path("/kaggle/working/rcq")
HF_HOME = Path("/kaggle/working/hf_cache")
RCQ_OUT = Path("/kaggle/working/outputs/qwen36_single_layer_rcq_pilot")
RCQ_LOG_DIR = Path("/kaggle/working/rcq_logs")

for path in [HF_HOME, RCQ_OUT.parent, RCQ_LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME)

print("=== selected commit ===", flush=True)
commit = run(f"git ls-remote {RCQ_REPO_URL} refs/heads/main").stdout.split()[0]
print(commit, flush=True)

print("=== gpu ===", flush=True)
run("nvidia-smi", check=False)

print("=== clone repo ===", flush=True)
run(f"rm -rf {RCQ_ROOT}")
run(f"git clone {RCQ_REPO_URL} {RCQ_ROOT}")
run(f"git checkout {commit}", cwd=RCQ_ROOT)
run("git rev-parse HEAD", cwd=RCQ_ROOT)

print("=== ensure qwen3.5 moe transformers support ===", flush=True)
run('python -m pip install -U "transformers>=5.9.0,<6"', cwd=RCQ_ROOT)
run(
    """python - <<'PY'
import transformers
print("transformers", transformers.__version__)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
print("qwen3_5_moe_decoder_layer_import", Qwen3_5MoeDecoderLayer.__name__)
PY""",
    cwd=RCQ_ROOT,
)

runner_log = RCQ_LOG_DIR / "qwen36_single_layer_rcq_pilot.runner.log"
resume_file = RCQ_OUT / "ablation_metrics.partial.json"
resume_arg = f" --skip-completed-from {resume_file}" if resume_file.exists() else ""
if resume_arg:
    print(f"=== resume from {resume_file} ===", flush=True)
cmd = f"""
PYTHONUNBUFFERED=1 PYTHONPATH={RCQ_ROOT} python -u scripts/qwen36_single_layer_rcq_ablation.py \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --layer-id 0 \
  --calib-docs 32 \
  --eval-docs 8 \
  --max-tokens-per-doc 1024 \
  --activation-source true_layer0_post_attention_norm \
  --max-experiments 13 \
  --verbose \
  --output-dir {RCQ_OUT} \
  --cache-dir {HF_HOME} \
  {resume_arg}
"""

print("=== run pilot ablations ===", flush=True)
with open(runner_log, "w", encoding="utf-8") as log:
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=RCQ_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )

last_size = 0
while proc.poll() is None:
    time.sleep(20)
    if runner_log.exists():
        text = runner_log.read_text(errors="replace")
        print(text[last_size:], end="", flush=True)
        last_size = len(text)
    print(f"\n=== still running {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===", flush=True)

text = runner_log.read_text(errors="replace") if runner_log.exists() else ""
print(text[last_size:], end="", flush=True)
print(f"\nrunner_exit_status={proc.returncode}", flush=True)
if proc.returncode != 0:
    raise RuntimeError("runner failed")

print("=== compact pilot summary ===", flush=True)
run(
    f"""python - <<'PY'
import json
p = "{RCQ_OUT}/ablation_metrics.json"
d = json.load(open(p))
print("status", d["status"])
print("activation_source", d["activation_source"])
print("activation_source_detail", d["activation_source_detail"])
print("doc_policy", d["doc_policy"])
print("elapsed_sec", d["elapsed_sec"])
for row in d["results"]:
    if row["status"] != "ok":
        print(row["label"], row["status"], row.get("reason"))
        continue
    held = row.get("heldout", {{}})
    print(row["label"], "heldout_mse", held.get("mse"), "heldout_rmse", held.get("rmse"))
PY"""
)

print("=== package compact results ===", flush=True)
run(
    """cd /kaggle/working && tar -czf qwen36_single_layer_rcq_pilot_outputs.tgz \
  outputs/qwen36_single_layer_rcq_pilot/run_manifest.json \
  outputs/qwen36_single_layer_rcq_pilot/index_summary.json \
  outputs/qwen36_single_layer_rcq_pilot/ablation_metrics.json \
  rcq_logs/qwen36_single_layer_rcq_pilot.runner.log && \
  ls -lh qwen36_single_layer_rcq_pilot_outputs.tgz"""
)
```

## 8. Add NMSE To Existing Pilot Metrics

Use this after a pilot run has already produced:

```text
/kaggle/working/outputs/qwen36_single_layer_rcq_pilot/ablation_metrics.json
```

This does not rebuild quantized ablations. It reloads the same layer, streams
the same raw documents, recomputes only the full-precision reference MoE-output
energy for calibration and held-out splits, then writes an annotated metrics
file with `nmse_mean_square` and `nmse_variance`.

Paste this into a normal Kaggle Python cell.

```python
import os
import subprocess
import time
from pathlib import Path


def run(cmd, cwd=None, check=True):
    print(f"\n=== $ {cmd} ===", flush=True)
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")
    return result


RCQ_REPO_URL = "https://github.com/AarushCodes/rcq.git"
RCQ_ROOT = Path("/kaggle/working/rcq")
HF_HOME = Path("/kaggle/working/hf_cache")
SOURCE_METRICS = Path("/kaggle/working/outputs/qwen36_single_layer_rcq_pilot/ablation_metrics.json")
RCQ_OUT = Path("/kaggle/working/outputs/qwen36_single_layer_rcq_pilot_nmse")
RCQ_LOG_DIR = Path("/kaggle/working/rcq_logs")

if not SOURCE_METRICS.exists():
    raise FileNotFoundError(f"missing existing metrics JSON: {SOURCE_METRICS}")

for path in [HF_HOME, RCQ_OUT.parent, RCQ_LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME)

print("=== selected commit ===", flush=True)
commit = run(f"git ls-remote {RCQ_REPO_URL} refs/heads/main").stdout.split()[0]
print(commit, flush=True)

print("=== clone repo ===", flush=True)
run(f"rm -rf {RCQ_ROOT}")
run(f"git clone {RCQ_REPO_URL} {RCQ_ROOT}")
run(f"git checkout {commit}", cwd=RCQ_ROOT)
run("git rev-parse HEAD", cwd=RCQ_ROOT)

print("=== ensure qwen3.5 moe transformers support ===", flush=True)
run('python -m pip install -U "transformers>=5.9.0,<6"', cwd=RCQ_ROOT)
run(
    """python - <<'PY'
import transformers
print("transformers", transformers.__version__)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
print("qwen3_5_moe_decoder_layer_import", Qwen3_5MoeDecoderLayer.__name__)
PY""",
    cwd=RCQ_ROOT,
)

runner_log = RCQ_LOG_DIR / "qwen36_single_layer_rcq_pilot_nmse.runner.log"
cmd = f"""
PYTHONUNBUFFERED=1 PYTHONPATH={RCQ_ROOT} python -u scripts/qwen36_single_layer_rcq_ablation.py \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --layer-id 0 \
  --calib-docs 32 \
  --eval-docs 8 \
  --max-tokens-per-doc 1024 \
  --activation-source true_layer0_post_attention_norm \
  --denominator-only \
  --metrics-json {SOURCE_METRICS} \
  --verbose \
  --output-dir {RCQ_OUT} \
  --cache-dir {HF_HOME}
"""

print("=== compute NMSE denominators only ===", flush=True)
with open(runner_log, "w", encoding="utf-8") as log:
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=RCQ_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )

last_size = 0
while proc.poll() is None:
    time.sleep(20)
    if runner_log.exists():
        text = runner_log.read_text(errors="replace")
        print(text[last_size:], end="", flush=True)
        last_size = len(text)
    print(f"\n=== still running {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===", flush=True)

text = runner_log.read_text(errors="replace") if runner_log.exists() else ""
print(text[last_size:], end="", flush=True)
print(f"\nrunner_exit_status={proc.returncode}", flush=True)
if proc.returncode != 0:
    raise RuntimeError("runner failed")

print("=== compact NMSE summary ===", flush=True)
run(
    f"""python - <<'PY'
import json
p = "{RCQ_OUT}/ablation_metrics_with_nmse.json"
d = json.load(open(p))
print("status", d["status"])
print("reference_stats", d["reference_stats"])
for row in d["results"]:
    if row["status"] != "ok":
        print(row["label"], row["status"], row.get("reason"))
        continue
    held = row.get("heldout", {{}})
    print(
        row["label"],
        "heldout_mse", held.get("mse"),
        "heldout_nmse_mean_square", held.get("nmse_mean_square"),
        "heldout_nmse_variance", held.get("nmse_variance"),
    )
PY"""
)

print("=== package NMSE results ===", flush=True)
run(
    """cd /kaggle/working && tar -czf qwen36_single_layer_rcq_pilot_nmse_outputs.tgz \
  outputs/qwen36_single_layer_rcq_pilot_nmse/run_manifest.json \
  outputs/qwen36_single_layer_rcq_pilot_nmse/index_summary.json \
  outputs/qwen36_single_layer_rcq_pilot_nmse/nmse_denominators.json \
  outputs/qwen36_single_layer_rcq_pilot_nmse/ablation_metrics_with_nmse.json \
  rcq_logs/qwen36_single_layer_rcq_pilot_nmse.runner.log && \
  ls -lh qwen36_single_layer_rcq_pilot_nmse_outputs.tgz"""
)
```
