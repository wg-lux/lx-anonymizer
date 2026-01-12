import math

import pytest
from pydantic import BaseModel

from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


def test_init_normalizes_blanks_and_trims() -> None:
    meta = SensitiveMeta(
        patient_first_name="  John  ",
        patient_last_name="Doe",
        casenumber="n/a",
        patient_dob="",
        examination_time=float("nan"),
        text=[],
        anonymized_text={},
        endoscope_sn="UNKNOWN",
        extra_field="ignored",
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
