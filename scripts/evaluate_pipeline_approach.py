#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lx_anonymizer.evaluation.pipeline_evaluation import (
    evaluate_feedback_alignment,
    evaluate_pair_files,
    evaluate_study_dataset,
    load_records,
)


def _parse_fields(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ("first_name", "last_name")
    fields = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not fields:
        raise ValueError("At least one non-empty field is required.")
    return fields


def _parse_key_fields(raw: str) -> tuple[str, ...]:
    keys = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not keys:
        raise ValueError("At least one key field is required.")
    return keys


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _print_scenario_summary(result: dict) -> None:
    print(f"\nScenario: {result['scenario']}")
    print(
        "Records "
        f"matched={result['matched_records']}, "
        f"gold={result['total_gold_records']}, "
        f"pred={result['total_prediction_records']}, "
        f"missing={result['missing_in_predictions']}, "
        f"extra={result['extra_in_predictions']}"
    )
    print(
        "Macro "
        f"exact={_pct(result['macro_exact_match_rate'])}, "
        f"similarity={_pct(result['macro_similarity'])}, "
        f"presence_f1={_pct(result['macro_presence_f1'])}"
    )
    print("Field metrics:")

    header = (
        f"{'field':<16}"
        f"{'support':>8}"
        f"{'exact':>10}"
        f"{'sim':>10}"
        f"{'p':>10}"
        f"{'r':>10}"
        f"{'f1':>10}"
    )
    print(header)
    print("-" * len(header))

    for field_name, metric in result["field_metrics"].items():
        print(
            f"{field_name:<16}"
            f"{metric['support']:>8}"
            f"{_pct(metric['exact_match_rate']):>10}"
            f"{_pct(metric['mean_similarity']):>10}"
            f"{_pct(metric['presence_precision']):>10}"
            f"{_pct(metric['presence_recall']):>10}"
            f"{_pct(metric['presence_f1']):>10}"
        )


def _print_ranking(results: Sequence[dict]) -> None:
    ranked = sorted(results, key=lambda item: item["macro_similarity"], reverse=True)
    print("\nRanking (by macro similarity):")
    for idx, item in enumerate(ranked, start=1):
        print(
            f"{idx}. {item['scenario']} "
            f"(similarity={_pct(item['macro_similarity'])}, "
            f"exact={_pct(item['macro_exact_match_rate'])})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate metadata extraction quality for lx-anonymizer study data "
            "or custom prediction/gold pairs, with optional frontend feedback alignment."
        )
    )
    parser.add_argument(
        "--study-dir",
        default="study_data",
        help="Study dataset directory (default: study_data).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Custom predictions file (JSON array or JSONL/NDJSON).",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        help="Custom gold file (JSON array or JSONL/NDJSON).",
    )
    parser.add_argument(
        "--fields",
        help="Comma-separated fields to evaluate for custom pairs (default: first_name,last_name).",
    )
    parser.add_argument(
        "--key-fields",
        default="file,report_id",
        help="Comma-separated key fields for record matching (default: file,report_id).",
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        help=(
            "Optional frontend feedback file with did_anonymization_change labels. "
            "Used as baseline alignment metric."
        ),
    )
    parser.add_argument(
        "--feedback-predictions",
        type=Path,
        help=(
            "Predictions file to compare with feedback labels. "
            "Defaults to --predictions when provided."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path to write full JSON report.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    key_fields = _parse_key_fields(args.key_fields)

    if bool(args.predictions) != bool(args.gold):
        parser.error("Use --predictions and --gold together.")

    if args.predictions and args.gold:
        fields = _parse_fields(args.fields)
        scenario_eval = evaluate_pair_files(
            prediction_path=args.predictions,
            gold_path=args.gold,
            fields=fields,
            scenario=f"{args.predictions.name} vs {args.gold.name}",
            key_fields=key_fields,
        )
        evaluations = [scenario_eval]
    else:
        evaluations = evaluate_study_dataset(args.study_dir)

    evaluation_dicts = [entry.to_dict() for entry in evaluations]

    for result in evaluation_dicts:
        _print_scenario_summary(result)

    _print_ranking(evaluation_dicts)

    payload = {"scenarios": evaluation_dicts}

    if args.feedback:
        predictions_for_feedback = args.feedback_predictions or args.predictions
        if predictions_for_feedback is None:
            parser.error(
                "--feedback requires --feedback-predictions when running in study mode."
            )

        prediction_records = load_records(predictions_for_feedback)
        feedback_records = load_records(args.feedback)
        feedback_eval = evaluate_feedback_alignment(
            predictions=prediction_records,
            feedback=feedback_records,
            key_fields=key_fields,
        )
        payload["feedback_alignment"] = feedback_eval.to_dict()

        print("\nFrontend feedback alignment:")
        print(
            f"pairs={feedback_eval.evaluated_pairs}, "
            f"accuracy={_pct(feedback_eval.accuracy)}, "
            f"precision={_pct(feedback_eval.precision)}, "
            f"recall={_pct(feedback_eval.recall)}, "
            f"f1={_pct(feedback_eval.f1)}"
        )
        print(
            f"confusion: tp={feedback_eval.tp}, tn={feedback_eval.tn}, "
            f"fp={feedback_eval.fp}, fn={feedback_eval.fn}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
