from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcq_moe.text_data import (
    TextFixtureConfig,
    load_hf_stream_texts,
    prepare_documents,
    split_calib_eval,
    synthetic_fixture_documents,
    write_text_fixture,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local calibration/eval text fixtures.")
    parser.add_argument("--source", choices=["synthetic", "hf"], default="synthetic")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-docs", type=int, default=256)
    parser.add_argument("--max-chars-per-doc", type=int, default=2000)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=Path("data/text_fixtures/generated"))
    parser.add_argument("--prefix", default="fineweb_edu")
    args = parser.parse_args()

    config = TextFixtureConfig(
        max_docs=args.max_docs,
        max_chars_per_doc=args.max_chars_per_doc,
        eval_fraction=args.eval_fraction,
        min_chars=args.min_chars,
    )
    raw_texts = synthetic_fixture_documents() if args.source == "synthetic" else load_hf_stream_texts(
        args.dataset,
        name=args.name,
        split=args.split,
        text_field=args.text_field,
    )
    docs = prepare_documents(raw_texts, config)
    calib_docs, eval_docs = split_calib_eval(docs, args.eval_fraction)

    calib_path = args.output_dir / f"{args.prefix}_calib.txt"
    eval_path = args.output_dir / f"{args.prefix}_eval.txt"
    write_text_fixture(calib_docs, calib_path, title=f"{args.prefix} calibration fixture")
    write_text_fixture(eval_docs, eval_path, title=f"{args.prefix} eval fixture")

    print("TEXT FIXTURE BUILD")
    print(f"source={args.source}")
    if args.source == "hf":
        print(f"dataset={args.dataset} name={args.name} split={args.split} text_field={args.text_field}")
    print(f"docs_total={len(docs)} docs_calib={len(calib_docs)} docs_eval={len(eval_docs)}")
    print(f"calib_path={calib_path}")
    print(f"eval_path={eval_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    if args.source == "hf":
        # Some HF streaming backends can leave non-daemon network/cache workers
        # alive after partial iteration. This script has already written files.
        os._exit(0)


if __name__ == "__main__":
    main()
