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

exec > >(tee -a "$RCQ_LOG_DIR/bootstrap.log") 2>&1

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

echo "=== dry-run inspect ==="
PYTHONPATH="$RCQ_ROOT" python scripts/qwen36_single_layer_rcq_ablation.py \
  --dry-run \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --layer-id 0 \
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

exec > >(tee -a "$RCQ_LOG_DIR/qwen36_single_layer_rcq_ablation.log") 2>&1

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

echo "=== run ablations ==="
PYTHONPATH="$RCQ_ROOT" python scripts/qwen36_single_layer_rcq_ablation.py \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --layer-id 0 \
  --calib-docs 256 \
  --eval-docs 64 \
  --max-tokens-per-doc 4096 \
  --activation-source true_layer0_post_attention_norm \
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
