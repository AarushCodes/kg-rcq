from rcq_moe.text_data import TextFixtureConfig, normalize_text, prepare_documents, split_calib_eval, synthetic_fixture_documents


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
