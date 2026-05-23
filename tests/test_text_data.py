import torch

from rcq_moe.text_data import (
    TextBatchConfig,
    TextFixtureConfig,
    encode_texts_as_toy_byte_tokens,
    normalize_text,
    prepare_documents,
    read_text_fixture,
    split_calib_eval,
    synthetic_fixture_documents,
    texts_to_input_ids,
    write_text_fixture,
)


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
