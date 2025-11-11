# lx_anonymizer/frame_cleaner/sensitive_meta_interface.py
from dataclasses import dataclass, asdict
from typing import Any, Optional

from django.template.defaultfilters import first
from .text_anonymizer import anonymize_text

@dataclass
class SensitiveMeta:
    file_path: Optional[str] = None
    patient_first_name: Optional[str] = None
    patient_last_name: Optional[str] = None
    patient_dob: Optional[str] = None
    casenumber: Optional[str] = None
    patient_gender: Optional[str] = None
    examination_date: Optional[str] = None
    examination_time: Optional[str] = None
    examiner_first_name: Optional[str] = None
    examiner_last_name: Optional[str] = None
    center: Optional[str] = None
    text: Optional[str]=None
    anonymized_text: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Invalid key '{key}' for SensitiveMeta")

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Invalid key '{key}' for SensitiveMeta")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensitiveMeta":
        """Safe from_dict with ignored unknown or missing fields."""
        if not data.get("anonymized_text") and data.get("first_name") and data.get("last_name") and data.get("text"):
            first_name = data.get("first_name")
            last_name = data.get("last_name")
            data["anonymized_text"] = anonymize_text(
                report_meta=data,
                text=data["text"],
                first_names=[first_name],
                last_names=[last_name],
            )
        valid = {k: v for k, v in (data or {}).items() if k in cls.__annotations__}
        return cls(**valid)

