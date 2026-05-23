import torch

from rcq_moe.harness import run_official_qwen_harness
from rcq_moe.official_qwen import make_tiny_official_qwen35_moe
from rcq_moe.quantization import RescueConfig
from rcq_moe.text_data import TextBatchConfig, split_calib_eval, synthetic_fixture_documents, texts_to_toy_token_batch


def test_official_qwen_harness_runs_on_text_derived_toy_tokens():
    torch.manual_seed(31)
    docs = synthetic_fixture_documents()
    calib_docs, eval_docs = split_calib_eval(docs, 0.25)
    model = make_tiny_official_qwen35_moe(
        vocab_size=64,
        hidden_size=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_hidden_layers=1,
    )
    calibration = texts_to_toy_token_batch(
        calib_docs,
        vocab_size=model.config.vocab_size,
        config=TextBatchConfig(batch_size=3, sequence_length=8, max_batches=1, repeat_if_needed=True),
    )
    eval_batch = texts_to_toy_token_batch(
        eval_docs,
        vocab_size=model.config.vocab_size,
        config=TextBatchConfig(batch_size=2, sequence_length=8, max_batches=1, repeat_if_needed=True),
    )

    result = run_official_qwen_harness(
        model,
        calibration.input_ids,
        eval_batch.input_ids,
        RescueConfig.rcq_1p75(block_size=8),
        calibration_attention_mask=calibration.attention_mask,
        eval_attention_mask=eval_batch.attention_mask,
        rank=4,
        fit_correction=True,
    )

    assert calibration.input_ids.shape == (3, 8)
    assert eval_batch.input_ids.shape == (2, 8)
    assert all(torch.isfinite(torch.tensor(value)) for value in result.kl.values())
    assert len(result.layer_diagnostics) == 1
