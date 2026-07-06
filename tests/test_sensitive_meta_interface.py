import math
from datetime import date
from typing import Mapping, cast

import pytest
from pydantic import BaseModel
from pydantic import ValidationError

from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


def test_init_normalizes_blanks_and_trims() -> None:
    meta = SensitiveMeta.model_validate(
        {
            "first_name": "  John  ",
            "last_name": "Doe",
            "casenumber": "n/a",
            "dob": "",
            "examination_time": float("nan"),
            "text": [],
            "anonymized_text": {},
            "endoscope_sn": "UNKNOWN",
            "extra_field": "ignored",
        }
    )

    assert meta.first_name == "John"
    assert meta.last_name == "Doe"
    assert meta.casenumber is None
    assert meta.dob is None
    assert meta.examination_time is None
    assert meta.text is None
    assert meta.anonymized_text is None
    assert meta.endoscope_sn is None


def test_validate_assignment_normalizes_on_set() -> None:
    meta = SensitiveMeta(first_name="Alice")
    meta.last_name = "  Smith  "
    assert meta.last_name == "Smith"

    meta.last_name = "   "
    assert meta.last_name == "unknown"


def test_safe_update_does_not_downgrade() -> None:
    meta = SensitiveMeta(first_name="Alice")
    meta.safe_update(first_name="   ")
    assert meta.first_name == "Alice"


def test_safe_update_merges_and_ignores_extras() -> None:
    meta = SensitiveMeta(first_name="Alice", casenumber="123")
    meta.safe_update({"last_name": "Doe", "unknown_field": "x"})
    assert meta.first_name == "Alice"
    assert meta.last_name == "Doe"
    assert meta.casenumber == "123"


def test_safe_update_accepts_sensitive_meta() -> None:
    meta = SensitiveMeta(first_name="Alice")
    meta.safe_update(SensitiveMeta(casenumber="456"))
    assert meta.casenumber == "456"


def test_safe_update_accepts_base_model() -> None:
    class Dummy(BaseModel):
        dob: str | None = None

    meta = SensitiveMeta(first_name="Alice")
    meta.safe_update(Dummy(dob="1990-01-01"))
    assert meta.dob is not None
    assert meta.dob.isoformat() == "1990-01-01"


def test_safe_update_rejects_unsupported_type() -> None:
    meta = SensitiveMeta(first_name="Alice")
    meta.safe_update(cast(Mapping[str, object] | None, 123))
    assert meta.first_name == "Alice"


def test_dict_access_and_conversion() -> None:
    meta = SensitiveMeta()
    meta["first_name"] = "John"
    assert meta.first_name == "John"
    assert meta["first_name"] == "John"

    data = meta.to_dict()
    assert "first_name" in data
    assert data["first_name"] == "John"

    with pytest.raises(KeyError):
        _ = meta["unknown_key"]

    with pytest.raises(KeyError):
        meta["unknown_key"] = "value"


def test_from_dict_and_extra_input() -> None:
    meta = SensitiveMeta.from_dict({"first_name": "Bob", "extra_field": "ignored"})
    assert meta.first_name == "Bob"
    assert not hasattr(meta, "extra_field")


def test_nan_handling_on_init() -> None:
    meta = SensitiveMeta.model_validate({"examination_time": math.nan})
    assert meta.examination_time is None


def test_swaps_exam_and_birth_dates_when_order_is_invalid_on_init() -> None:
    meta = SensitiveMeta.model_validate(
        {
            "dob": "2024-02-15",
            "examination_date": "1994-03-21",
        }
    )
    assert meta.dob is not None
    assert meta.examination_date is not None
    assert meta.dob.isoformat() == "1994-03-21"
    assert meta.examination_date.isoformat() == "2024-02-15"


def test_safe_update_swaps_exam_and_birth_dates_when_order_is_invalid() -> None:
    meta = SensitiveMeta()
    meta.safe_update(
        {
            "dob": "15.02.2024",
            "examination_date": "21.03.1994",
        }
    )
    assert meta.dob is not None
    assert meta.examination_date is not None
    assert meta.dob.isoformat() == "1994-03-21"
    assert meta.examination_date.isoformat() == "2024-02-15"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("unknown", "unknown"),
        ("UNDEFINED", "unknown"),
        ("Null", "unknown"),
        (" none ", "unknown"),
        ("NA", "unknown"),
        ("-", "unknown"),
        (" value ", "value"),
    ],
)
def test_normalize_and_clean_handles_null_equivalents(
    raw: str, expected: str | None
) -> None:
    meta = SensitiveMeta(first_name=raw)
    assert meta.first_name == expected


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
    parsed = SensitiveMeta._parse_date_like(raw)  # pyright: ignore[reportPrivateUsage]
    assert parsed is not None
    assert parsed.isoformat() == expected


def test_parse_date_like_rejects_unrecognized_format() -> None:
    assert (
        SensitiveMeta._parse_date_like(  # pyright: ignore[reportPrivateUsage]
            "01-31-2024"
        )
        is None
    )


def test_parse_date_like_uses_cached_results_for_repeated_inputs() -> None:
    SensitiveMeta._parse_date_like_cached.cache_clear()  # pyright: ignore[reportPrivateUsage]
    first = SensitiveMeta._parse_date_like("2024-01-31")  # pyright: ignore[reportPrivateUsage]
    second = SensitiveMeta._parse_date_like("2024-01-31")  # pyright: ignore[reportPrivateUsage]
    assert first is second

    cache_info = SensitiveMeta._parse_date_like_cached.cache_info()  # pyright: ignore[reportPrivateUsage]
    assert cache_info.hits >= 1


def test_date_order_rejects_unparseable_dates() -> None:
    with pytest.raises(ValidationError):
        SensitiveMeta(
            dob=cast(date | None, "2024-01-31"),
            examination_date=cast(date | None, "not-a-date"),
        )


def test_type_validation_rejects_non_string_scalars_after_preprocessing() -> None:
    with pytest.raises(ValidationError):
        SensitiveMeta(first_name=cast(str, ["Alice"]))


def test_safe_update_is_validation_gated_and_prevents_partial_mutation() -> None:
    meta = SensitiveMeta(first_name="Alice", last_name="Smith")
    meta.safe_update({"first_name": "Bob", "last_name": ["bad"]})

    # Payload is validated as a whole before assignment, so no field should change.
    assert meta.first_name == "Alice"
    assert meta.last_name == "Smith"


def test_safe_update_fill_only_keeps_existing_nonblank_values() -> None:
    meta = SensitiveMeta.model_validate(
        {"dob": "2000-01-01", "examination_date": "2020-01-01"}
    )
    meta.safe_update({"dob": "2024-01-01"})
    assert meta.dob is not None
    assert meta.examination_date is not None
    assert meta.dob.isoformat() == "2000-01-01"
    assert meta.examination_date.isoformat() == "2020-01-01"


def test_safe_update_bypasses_instance_setattr_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = SensitiveMeta(first_name="Alice")
    target_id = id(meta)
    original_setattr = SensitiveMeta.__setattr__
    trap_calls = {"count": 0}

    def trap(self: SensitiveMeta, name: str, value: object) -> None:
        if id(self) == target_id and name in SensitiveMeta.model_fields:
            trap_calls["count"] += 1
            raise AssertionError("safe_update should bypass SensitiveMeta.__setattr__")
        original_setattr(self, name, value)

    monkeypatch.setattr(SensitiveMeta, "__setattr__", trap)

    meta.safe_update({"last_name": "Doe"})

    assert trap_calls["count"] == 0
    assert meta.last_name == "Doe"
