from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from rapidfuzz import fuzz, utils


TITLE_WORDS = {
    "herr",
    "frau",
    "fru",
    "fruis",
    "fruher",
    "señor",
    "señorita",
    "mr.",
    "mrs.",
    "prof.",
    "prof",
    "doctor",
    "dr.",
    "dr",
    "sir",
    "madam",
    "monsieur",
    "herrn",
    "ing.",
    "ing",
    "pr.",
}

NAME_FIELDS = {
    "first_name",
    "last_name",
    "patient_first_name",
    "patient_last_name",
    "examiner_first_name",
    "examiner_last_name",
}

EMPTY_TOKENS = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "nan",
    "undefined",
    "unknown",
    "-",
}


@dataclass
class FieldMetrics:
    field: str
    support: int
    exact_match_rate: float
    mean_similarity: float
    presence_precision: float
    presence_recall: float
    presence_f1: float
    gold_non_empty: int
    predicted_non_empty: int


@dataclass
class ScenarioEvaluation:
    scenario: str
    key_fields: tuple[str, ...]
    total_gold_records: int
    total_prediction_records: int
    matched_records: int
    missing_in_predictions: int
    extra_in_predictions: int
    duplicate_gold_keys: int
    duplicate_prediction_keys: int
    macro_exact_match_rate: float
    macro_similarity: float
    macro_presence_f1: float
    field_metrics: Dict[str, FieldMetrics]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["field_metrics"] = {
            field: asdict(metrics) for field, metrics in self.field_metrics.items()
        }
        return data


@dataclass
class FeedbackAgreement:
    evaluated_pairs: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    text = str(value).strip()
    if not text:
        return ""
    if text.casefold() in EMPTY_TOKENS:
        return ""
    return text


def _strip_titles(name: str) -> str:
    if not name:
        return ""
    processed = utils.default_process(name)
    if not processed:
        return ""
    tokens = [token for token in processed.split() if token not in TITLE_WORDS]
    return " ".join(tokens)


def _canonical_value(field: str, value: Any) -> str:
    normalized = _normalize_scalar(value)
    if not normalized:
        return ""

    if field in NAME_FIELDS:
        return _strip_titles(normalized)

    processed = utils.default_process(normalized)
    return processed if processed is not None else normalized.casefold()


def _similarity(field: str, predicted: str, expected: str) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0

    if field in NAME_FIELDS:
        return fuzz.token_sort_ratio(predicted, expected) / 100.0

    return fuzz.ratio(predicted, expected) / 100.0


def _record_key(
    record: Mapping[str, Any],
    key_fields: Sequence[str],
    index_fallback: int,
) -> tuple[str, ...]:
    parts = tuple(_normalize_scalar(record.get(field)) for field in key_fields)
    if all(parts):
        return parts
    return (f"__idx__:{index_fallback}",)


def _build_record_map(
    records: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
) -> tuple[Dict[tuple[str, ...], Mapping[str, Any]], int]:
    record_map: Dict[tuple[str, ...], Mapping[str, Any]] = {}
    duplicate_keys = 0
    for index, record in enumerate(records):
        key = _record_key(record, key_fields=key_fields, index_fallback=index)
        if key in record_map:
            duplicate_keys += 1
        record_map[key] = record
    return record_map, duplicate_keys


def _parse_json_lines(raw_text: str, source_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        candidate = line[:-1].rstrip() if line.endswith(",") else line
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse JSONL line {line_number} in {source_path}: {exc.msg}"
            ) from exc

        if isinstance(parsed, dict):
            records.append(parsed)
            continue

        raise ValueError(f"Line {line_number} in {source_path} is not a JSON object.")

    return records


def load_records(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    raw_text = source.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    if raw_text.startswith("["):
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array in {source}, got {type(parsed)!r}")
        return [entry for entry in parsed if isinstance(entry, dict)]

    if raw_text.startswith("{") and "\n" not in raw_text:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return [parsed]
        raise ValueError(f"Unsupported JSON root in {source}: {type(parsed)!r}")

    if raw_text.startswith("{") and "\n" in raw_text:
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                nested_dict_values = all(
                    isinstance(value, dict) for value in parsed.values()
                )
                if nested_dict_values:
                    return [
                        {"report_id": str(key), **value}
                        for key, value in parsed.items()
                    ]
                return [parsed]
        except json.JSONDecodeError:
            pass

    return _parse_json_lines(raw_text, source)


def evaluate_records(
    predictions: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    scenario: str,
    key_fields: Sequence[str] = ("file", "report_id"),
) -> ScenarioEvaluation:
    prediction_map, duplicate_prediction_keys = _build_record_map(
        predictions, key_fields
    )
    gold_map, duplicate_gold_keys = _build_record_map(gold, key_fields)

    matched_keys = sorted(set(prediction_map) & set(gold_map))
    missing_keys = set(gold_map) - set(prediction_map)
    extra_keys = set(prediction_map) - set(gold_map)

    field_results: Dict[str, FieldMetrics] = {}

    for field in fields:
        support = 0
        exact_matches = 0
        similarity_sum = 0.0

        tp = 0
        fp = 0
        fn = 0
        gold_non_empty = 0
        prediction_non_empty = 0

        for key in matched_keys:
            prediction_record = prediction_map[key]
            gold_record = gold_map[key]

            predicted_value = _canonical_value(field, prediction_record.get(field))
            gold_value = _canonical_value(field, gold_record.get(field))

            predicted_non_empty = bool(predicted_value)
            gold_is_non_empty = bool(gold_value)

            if predicted_non_empty:
                prediction_non_empty += 1
            if gold_is_non_empty:
                gold_non_empty += 1

            if predicted_non_empty and gold_is_non_empty:
                tp += 1
            elif predicted_non_empty and not gold_is_non_empty:
                fp += 1
            elif not predicted_non_empty and gold_is_non_empty:
                fn += 1

            if predicted_non_empty or gold_is_non_empty:
                support += 1
                if predicted_value == gold_value:
                    exact_matches += 1
                similarity_sum += _similarity(field, predicted_value, gold_value)

        exact_match_rate = _safe_div(exact_matches, support)
        mean_similarity = _safe_div(similarity_sum, support)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        presence_f1 = _safe_div(2 * precision * recall, precision + recall)

        field_results[field] = FieldMetrics(
            field=field,
            support=support,
            exact_match_rate=exact_match_rate,
            mean_similarity=mean_similarity,
            presence_precision=precision,
            presence_recall=recall,
            presence_f1=presence_f1,
            gold_non_empty=gold_non_empty,
            predicted_non_empty=prediction_non_empty,
        )

    macro_exact = _safe_div(
        sum(metric.exact_match_rate for metric in field_results.values()),
        len(field_results),
    )
    macro_similarity = _safe_div(
        sum(metric.mean_similarity for metric in field_results.values()),
        len(field_results),
    )
    macro_presence_f1 = _safe_div(
        sum(metric.presence_f1 for metric in field_results.values()),
        len(field_results),
    )

    return ScenarioEvaluation(
        scenario=scenario,
        key_fields=tuple(key_fields),
        total_gold_records=len(gold),
        total_prediction_records=len(predictions),
        matched_records=len(matched_keys),
        missing_in_predictions=len(missing_keys),
        extra_in_predictions=len(extra_keys),
        duplicate_gold_keys=duplicate_gold_keys,
        duplicate_prediction_keys=duplicate_prediction_keys,
        macro_exact_match_rate=macro_exact,
        macro_similarity=macro_similarity,
        macro_presence_f1=macro_presence_f1,
        field_metrics=field_results,
    )


def evaluate_pair_files(
    prediction_path: Path | str,
    gold_path: Path | str,
    fields: Sequence[str],
    scenario: str | None = None,
    key_fields: Sequence[str] = ("file", "report_id"),
) -> ScenarioEvaluation:
    prediction_records = load_records(prediction_path)
    gold_records = load_records(gold_path)
    scenario_name = (
        scenario or f"{Path(prediction_path).name} vs {Path(gold_path).name}"
    )

    return evaluate_records(
        predictions=prediction_records,
        gold=gold_records,
        fields=fields,
        scenario=scenario_name,
        key_fields=key_fields,
    )


def evaluate_study_dataset(study_dir: Path | str) -> list[ScenarioEvaluation]:
    base = Path(study_dir)

    scenarios: list[dict[str, object]] = [
        {
            "name": "control_vs_gold",
            "pred": base / "names_control.json",
            "gold": base / "gold.json",
            "fields": ("first_name", "last_name"),
        },
        {
            "name": "regex_vs_gold_regex",
            "pred": base / "names_regex.json",
            "gold": base / "gold_regex.json",
            "fields": ("first_name", "last_name"),
        },
        {
            "name": "deepseek_vs_gold_deepseek",
            "pred": base / "names_deepseek.json",
            "gold": base / "gold_deepseek.json",
            "fields": ("first_name", "last_name", "dob", "gender", "casenumber"),
        },
    ]

    results: list[ScenarioEvaluation] = []
    for scenario in scenarios:
        pred_path = scenario["pred"]
        gold_path = scenario["gold"]
        fields = scenario["fields"]
        scenario_name = scenario["name"]
        if not isinstance(pred_path, Path) or not isinstance(gold_path, Path):
            continue
        if not isinstance(fields, tuple) or not all(
            isinstance(field, str) for field in fields
        ):
            continue
        if not isinstance(scenario_name, str):
            continue
        if not pred_path.exists() or not gold_path.exists():
            continue

        result = evaluate_pair_files(
            prediction_path=pred_path,
            gold_path=gold_path,
            fields=fields,
            scenario=scenario_name,
            key_fields=("file", "report_id"),
        )
        results.append(result)

    if not results:
        raise FileNotFoundError(
            f"No known study-data evaluation pairs found in: {base}"
        )

    return results


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(int(value))

    normalized = _normalize_scalar(value).casefold()
    if not normalized:
        return None

    if normalized in {"1", "true", "yes", "y", "changed"}:
        return True
    if normalized in {"0", "false", "no", "n", "unchanged"}:
        return False

    return None


def infer_anonymization_change(record: Mapping[str, Any]) -> bool | None:
    explicit_keys = (
        "predicted_changed",
        "predicted_change",
        "did_anonymization_change",
        "anonymization_changed",
    )

    for key in explicit_keys:
        if key in record:
            parsed = _parse_bool(record.get(key))
            if parsed is not None:
                return parsed

    text = record.get("text")
    anonymized_text = record.get("anonymized_text")

    if isinstance(text, str) and isinstance(anonymized_text, str):
        canonical_text = " ".join(text.split())
        canonical_anonymized = " ".join(anonymized_text.split())
        return canonical_text != canonical_anonymized

    return None


def _iter_feedback_pairs(
    predictions: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    prediction_map, _ = _build_record_map(predictions, key_fields)
    feedback_map, _ = _build_record_map(feedback, key_fields)

    matched_keys = set(prediction_map) & set(feedback_map)
    if matched_keys:
        for key in sorted(matched_keys):
            yield prediction_map[key], feedback_map[key]
        return

    for prediction_record, feedback_record in zip(predictions, feedback):
        yield prediction_record, feedback_record


def evaluate_feedback_alignment(
    predictions: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str] = ("file", "report_id"),
) -> FeedbackAgreement:
    tp = tn = fp = fn = 0
    evaluated_pairs = 0

    label_keys = (
        "did_anonymization_change",
        "label",
        "frontend_label",
        "changed",
    )

    for prediction_record, feedback_record in _iter_feedback_pairs(
        predictions=predictions,
        feedback=feedback,
        key_fields=key_fields,
    ):
        predicted_label = infer_anonymization_change(prediction_record)

        feedback_label = None
        for key in label_keys:
            if key in feedback_record:
                feedback_label = _parse_bool(feedback_record.get(key))
                if feedback_label is not None:
                    break

        if predicted_label is None or feedback_label is None:
            continue

        evaluated_pairs += 1

        if predicted_label and feedback_label:
            tp += 1
        elif (not predicted_label) and (not feedback_label):
            tn += 1
        elif predicted_label and (not feedback_label):
            fp += 1
        else:
            fn += 1

    accuracy = _safe_div(tp + tn, evaluated_pairs)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return FeedbackAgreement(
        evaluated_pairs=evaluated_pairs,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )
