import math

import pytest
from pydantic import BaseModel
from pydantic import ValidationError

from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


def test_init_normalizes_blanks_and_trims() -> None:
    meta = SensitiveMeta.model_validate(
        {
            "patient_first_name": "  John  ",
            "patient_last_name": "Doe",
            "casenumber": "n/a",
            "patient_dob": "",
            "examination_time": float("nan"),
            "text": [],
            "anonymized_text": {},
            "endoscope_sn": "UNKNOWN",
            "extra_field": "ignored",
        }
    )

    assert meta.patient_first_name == "John"
    assert meta.patient_last_name == "Doe"
    assert meta.casenumber is None
    assert meta.patient_dob is None
    assert meta.examination_time is None
    assert meta.text is None
    assert meta.anonymized_text is None
    assert meta.endoscope_sn is None


def test_validate_assignment_normalizes_on_set() -> None:
    meta = SensitiveMeta(patient_first_name="Alice")
    meta.patient_last_name = "  Smith  "
    assert meta.patient_last_name == "Smith"

    meta.patient_last_name = "   "
    assert meta.patient_last_name is None


def test_safe_update_does_not_downgrade() -> None:
    meta = SensitiveMeta(patient_first_name="Alice")
    meta.safe_update(patient_first_name="   ")
    assert meta.patient_first_name == "Alice"


def test_safe_update_merges_and_ignores_extras() -> None:
    meta = SensitiveMeta(patient_first_name="Alice", casenumber="123")
    meta.safe_update({"patient_last_name": "Doe", "unknown_field": "x"})
    assert meta.patient_first_name == "Alice"
    assert meta.patient_last_name == "Doe"
    assert meta.casenumber == "123"


def test_safe_update_accepts_sensitive_meta() -> None:
    meta = SensitiveMeta(patient_first_name="Alice")
    meta.safe_update(SensitiveMeta(casenumber="456"))
    assert meta.casenumber == "456"


def test_safe_update_accepts_base_model() -> None:
    class Dummy(BaseModel):
        patient_dob: str | None = None

    meta = SensitiveMeta(patient_first_name="Alice")
    meta.safe_update(Dummy(patient_dob="1990-01-01"))
    assert meta.patient_dob == "1990-01-01"


def test_safe_update_rejects_unsupported_type() -> None:
    meta = SensitiveMeta(patient_first_name="Alice")
    meta.safe_update(123)
    assert meta.patient_first_name == "Alice"


def test_dict_access_and_conversion() -> None:
    meta = SensitiveMeta()
    meta["patient_first_name"] = "John"
    assert meta.patient_first_name == "John"
    assert meta["patient_first_name"] == "John"

    data = meta.to_dict()
    assert "patient_first_name" in data
    assert data["patient_first_name"] == "John"

    with pytest.raises(KeyError):
        _ = meta["unknown_key"]

    with pytest.raises(KeyError):
        meta["unknown_key"] = "value"


def test_from_dict_and_extra_input() -> None:
    meta = SensitiveMeta.from_dict(
        {"patient_first_name": "Bob", "extra_field": "ignored"}
    )
    assert meta.patient_first_name == "Bob"
    assert not hasattr(meta, "extra_field")


def test_nan_handling_on_init() -> None:
    meta = SensitiveMeta(examination_time=math.nan)
    assert meta.examination_time is None


def test_swaps_exam_and_birth_dates_when_order_is_invalid_on_init() -> None:
    meta = SensitiveMeta(
        patient_dob="2024-02-15",
        examination_date="1994-03-21",
    )
    assert meta.patient_dob == "1994-03-21"
    assert meta.examination_date == "2024-02-15"


def test_safe_update_swaps_exam_and_birth_dates_when_order_is_invalid() -> None:
    meta = SensitiveMeta()
    meta.safe_update(
        {
            "patient_dob": "15.02.2024",
            "examination_date": "21.03.1994",
        }
    )
    assert meta.patient_dob == "21.03.1994"
    assert meta.examination_date == "15.02.2024"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("unknown", None),
        ("UNDEFINED", None),
        ("Null", None),
        (" none ", None),
        ("NA", None),
        ("-", None),
        (" value ", "value"),
    ],
)
def test_normalize_and_clean_handles_null_equivalents(
    raw: str, expected: str | None
) -> None:
    meta = SensitiveMeta(patient_first_name=raw)
    assert meta.patient_first_name == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-01-31", "2024-01-31"),
        ("31.01.2024", "2024-01-31"),
        ("31.01.24", "2024-01-31"),
        ("31/01/2024", "2024-01-31"),
        ("31-01-2024", "2024-01-31"),
        ("2024/01/31", "2024-01-31"),
    ],
)
def test_parse_date_like_supports_all_configured_formats(
    raw: str, expected: str
) -> None:
    parsed = SensitiveMeta._parse_date_like(raw)
    assert parsed is not None
    assert parsed.isoformat() == expected


def test_parse_date_like_rejects_unrecognized_format() -> None:
    assert SensitiveMeta._parse_date_like("01-31-2024") is None


def test_parse_date_like_uses_cached_results_for_repeated_inputs() -> None:
    SensitiveMeta._parse_date_like_cached.cache_clear()
    first = SensitiveMeta._parse_date_like("2024-01-31")
    second = SensitiveMeta._parse_date_like("2024-01-31")
    assert first is second

    cache_info = SensitiveMeta._parse_date_like_cached.cache_info()
    assert cache_info.hits >= 1


def test_date_order_swap_only_applies_when_both_dates_are_parseable() -> None:
    meta = SensitiveMeta(patient_dob="2024-01-31", examination_date="not-a-date")
    assert meta.patient_dob == "2024-01-31"
    assert meta.examination_date == "not-a-date"


def test_type_validation_rejects_non_string_scalars_after_preprocessing() -> None:
    with pytest.raises(ValidationError):
        SensitiveMeta(patient_first_name=["Alice"])


def test_safe_update_is_validation_gated_and_prevents_partial_mutation() -> None:
    meta = SensitiveMeta(patient_first_name="Alice", patient_last_name="Smith")
    meta.safe_update({"patient_first_name": "Bob", "patient_last_name": ["bad"]})

    # Payload is validated as a whole before assignment, so no field should change.
    assert meta.patient_first_name == "Alice"
    assert meta.patient_last_name == "Smith"


def test_safe_update_fill_only_keeps_existing_nonblank_values() -> None:
    meta = SensitiveMeta(patient_dob="2000-01-01", examination_date="2020-01-01")
    meta.safe_update({"patient_dob": "2024-01-01"})
    assert meta.patient_dob == "2000-01-01"
    assert meta.examination_date == "2020-01-01"


def test_safe_update_bypasses_instance_setattr_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = SensitiveMeta(patient_first_name="Alice")
    target_id = id(meta)
    original_setattr = SensitiveMeta.__setattr__
    trap_calls = {"count": 0}

    def trap(self: SensitiveMeta, name: str, value: object) -> None:
        if id(self) == target_id and name in SensitiveMeta.model_fields:
            trap_calls["count"] += 1
            raise AssertionError("safe_update should bypass SensitiveMeta.__setattr__")
        original_setattr(self, name, value)

    monkeypatch.setattr(SensitiveMeta, "__setattr__", trap)

    meta.safe_update({"patient_last_name": "Doe"})

    assert trap_calls["count"] == 0
    assert meta.patient_last_name == "Doe"
