# lx_anonymizer/frame_cleaner/sensitive_meta_interface.py
from dataclasses import dataclass, asdict
from typing import Any, Optional

@dataclass
class SensitiveMeta:
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

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Invalid key '{key}' for SensitiveMeta")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensitiveMeta":
        """Safe from_dict with ignored unknown fields."""
        valid = {k: v for k, v in (data or {}).items() if k in cls.__annotations__}
        return cls(**valid)
