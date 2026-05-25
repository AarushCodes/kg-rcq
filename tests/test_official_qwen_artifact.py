import json

import torch

from rcq_moe.artifact import load_official_qwen_rcq_artifact, save_official_qwen_rcq_artifact
from rcq_moe.official_qwen import OfficialQwen35MoeRCQSparseMoeBlock, convert_official_qwen35_moe_to_rcq_with_diagnostics, make_tiny_official_qwen35_moe
from rcq_moe.quantization import RescueConfig


def test_official_qwen_rcq_artifact_roundtrip_matches_converted_logits(tmp_path):
    torch.manual_seed(23)
    model = make_tiny_official_qwen35_moe(
        vocab_size=48,
        hidden_size=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_hidden_layers=1,
    )
    calibration_ids = torch.randint(0, model.config.vocab_size, (4, 5))
    eval_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    conversion = convert_official_qwen35_moe_to_rcq_with_diagnostics(
        model,
        calibration_ids,
        RescueConfig.rcq_1p75(block_size=8),
        rank=4,
        fit_correction=True,
    )

    save_official_qwen_rcq_artifact(conversion.model, tmp_path, diagnostics=conversion.layer_diagnostics)
    loaded = load_official_qwen_rcq_artifact(tmp_path)

    converted_logits = conversion.model(input_ids=eval_ids, use_cache=False).logits
    loaded_logits = loaded(input_ids=eval_ids, use_cache=False).logits

    assert isinstance(loaded.model.layers[0].mlp, OfficialQwen35MoeRCQSparseMoeBlock)
    assert torch.allclose(loaded_logits, converted_logits, atol=0.0, rtol=0.0)


def test_official_qwen_rcq_artifact_metadata_and_files(tmp_path):
    torch.manual_seed(24)
    model = make_tiny_official_qwen35_moe(vocab_size=32, hidden_size=8, moe_intermediate_size=8, num_hidden_layers=1)
    calibration_ids = torch.randint(0, model.config.vocab_size, (3, 4))
    conversion = convert_official_qwen35_moe_to_rcq_with_diagnostics(
        model,
        calibration_ids,
        RescueConfig.rcq_1p75(block_size=8),
        rank=4,
        fit_correction=True,
    )

    save_official_qwen_rcq_artifact(conversion.model, tmp_path, diagnostics=conversion.layer_diagnostics)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert (tmp_path / "non_expert_state.pt").exists()
    assert (tmp_path / "rcq_state.pt").exists()
    assert metadata["artifact_format"] == "rcq_official_qwen35_moe_v1"
    assert metadata["model_type"] == "Qwen3_5MoeForCausalLM"
    assert metadata["num_layers"] == 1
    assert metadata["diagnostics"][0]["linear_diagnostics"]["gate"]["bpw"] > 0.0


def test_official_qwen_rcq_artifact_roundtrip_preserves_down_none(tmp_path):
    torch.manual_seed(25)
    model = make_tiny_official_qwen35_moe(vocab_size=32, hidden_size=8, moe_intermediate_size=8, num_hidden_layers=1)
    calibration_ids = torch.randint(0, model.config.vocab_size, (3, 4))
    eval_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    conversion = convert_official_qwen35_moe_to_rcq_with_diagnostics(
        model,
        calibration_ids,
        RescueConfig.down_min2_5p4(block_size=8),
        rank=4,
        fit_correction=False,
        down_shared_mode="none",
        down_moment_mode="per_expert",
    )

    save_official_qwen_rcq_artifact(conversion.model, tmp_path, diagnostics=conversion.layer_diagnostics)
    loaded = load_official_qwen_rcq_artifact(tmp_path)

    converted_logits = conversion.model(input_ids=eval_ids, use_cache=False).logits
    loaded_logits = loaded(input_ids=eval_ids, use_cache=False).logits

    assert loaded.model.layers[0].mlp.experts.q_down.shared_mode == "none"
    assert torch.allclose(loaded_logits, converted_logits, atol=0.0, rtol=0.0)
