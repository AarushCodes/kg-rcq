import torch
from transformers import AutoTokenizer

from rcq_moe.text_data import (
    TextBatchConfig,
    TextFixtureConfig,
    encode_texts_as_toy_byte_tokens,
    normalize_text,
    prepare_documents,
    read_text_fixture,
    split_calib_eval,
    synthetic_fixture_documents,
    texts_to_hf_token_batch,
    texts_to_input_ids,
    texts_to_toy_token_batch,
    write_text_fixture,
)


TINY_TOKENIZER_DIR = "tests/fixtures/tiny_bert_tokenizer"


def test_normalize_text_collapses_whitespace_and_blank_lines():
    raw = "  alpha   beta\r\n\r\n gamma\t delta  "
    assert normalize_text(raw) == "alpha beta\ngamma delta"


def test_prepare_documents_filters_truncates_and_caps_count():
    raw = ["short", "a" * 100, "b" * 120, "c" * 140]
    docs = prepare_documents(raw, TextFixtureConfig(max_docs=2, max_chars_per_doc=50, min_chars=80))
    assert docs == ["a" * 50, "b" * 50]


def test_split_calib_eval_is_deterministic_and_nonempty():
    docs = [f"doc {idx}" for idx in range(10)]
    calib, eval_docs = split_calib_eval(docs, 0.2)
    assert calib == docs[:8]
    assert eval_docs == docs[8:]


def test_synthetic_fixture_has_enough_documents_for_offline_smoke():
    docs = prepare_documents(synthetic_fixture_documents(), TextFixtureConfig(max_docs=256, min_chars=20))
    calib, eval_docs = split_calib_eval(docs, 0.25)
    assert len(calib) >= 4
    assert len(eval_docs) >= 1


def test_read_text_fixture_roundtrips_written_documents(tmp_path):
    docs = ["alpha beta", "gamma\ndelta"]
    path = tmp_path / "fixture.txt"
    write_text_fixture(docs, path, title="toy")

    assert read_text_fixture(path) == docs


def test_toy_byte_tokens_are_deterministic_nonzero_and_in_vocab():
    ids = encode_texts_as_toy_byte_tokens(["abc", "abc"], vocab_size=16)
    assert ids == encode_texts_as_toy_byte_tokens(["abc", "abc"], vocab_size=16)
    assert min(ids) >= 1
    assert max(ids) < 16


def test_texts_to_input_ids_fixed_shape_with_explicit_repeat():
    input_ids = texts_to_input_ids(
        ["abc"],
        vocab_size=8,
        config=TextBatchConfig(batch_size=2, sequence_length=5, max_batches=3, repeat_if_needed=True),
    )

    assert input_ids.shape == (6, 5)
    assert input_ids.dtype == torch.long
    assert int(input_ids.min()) >= 1
    assert int(input_ids.max()) < 8


def test_toy_token_batch_returns_all_valid_attention_mask():
    batch = texts_to_toy_token_batch(
        ["abc"],
        vocab_size=8,
        config=TextBatchConfig(batch_size=1, sequence_length=6, max_batches=1, repeat_if_needed=True),
    )

    assert batch.input_ids.shape == (1, 6)
    assert batch.attention_mask.shape == (1, 6)
    assert torch.equal(batch.attention_mask, torch.ones_like(batch.attention_mask))


def test_hf_token_batch_pads_short_text_and_marks_attention_mask():
    class StubTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def encode(self, text, *, add_special_tokens):
            del add_special_tokens
            return [(ord(char) % 17) + 3 for char in text if not char.isspace()]

    batch = texts_to_hf_token_batch(
        ["abc"],
        tokenizer=StubTokenizer(),
        config=TextBatchConfig(batch_size=2, sequence_length=4, max_batches=1, pad_if_needed=True),
    )

    assert batch.input_ids.shape == (2, 4)
    assert batch.attention_mask.tolist() == [[1, 1, 1, 0], [0, 0, 0, 0]]
    assert batch.input_ids[0, 3].item() == 0


def test_hf_token_batch_loads_local_auto_tokenizer_without_network():
    tokenizer = AutoTokenizer.from_pretrained(TINY_TOKENIZER_DIR, local_files_only=True)
    batch = texts_to_hf_token_batch(
        ["Router coherent quantization for MoE experts."],
        tokenizer=tokenizer,
        config=TextBatchConfig(batch_size=1, sequence_length=10, max_batches=1, pad_if_needed=True),
    )

    assert len(tokenizer) == 34
    assert batch.input_ids.shape == (1, 10)
    assert batch.attention_mask.shape == (1, 10)
    assert int(batch.input_ids.max()) < len(tokenizer)
    assert 0 < int(batch.attention_mask.sum().item()) < 10
