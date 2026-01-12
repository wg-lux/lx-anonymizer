# lx_anonymizer/frame_cleaner/sensitive_meta_interface.py
import math
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from lx_anonymizer.setup.custom_logger import logger


class SensitiveMeta(BaseModel):
    """
    Metadata container for sensitive patient information.
    Handles safe updates and normalization of all sensitive fields, that will be needed in the anonymization process.
    Migrated to Pydantic to ensure automatic normalization and validation.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
    )

    file_path: Optional[str] = None
    patient_first_name: Optional[str] = None
    patient_last_name: Optional[str] = None
    patient_dob: Optional[str] = None
    casenumber: Optional[str] = None
    patient_gender_name: Optional[str] = None
    examination_date: Optional[str] = None
    examination_time: Optional[str] = None
    examiner_first_name: Optional[str] = None
    examiner_last_name: Optional[str] = None
    center: Optional[str] = None
    text: Optional[str] = None
    anonymized_text: Optional[str] = None
    endoscope_type: Optional[str] = None
    endoscope_sn: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def normalize_and_clean(cls, v: Any) -> Any:
        """
        Global validator that runs on all fields BEFORE type checking.
        Replaces legacy _is_blank and _normalize_scalar functions.
        """
        if v is None:
            return None

        if isinstance(v, float) and math.isnan(v):
            return None

        if isinstance(v, (list, dict, set, tuple)) and len(v) == 0:
            return None

        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            if s.casefold() in {"unknown", "undefined", "null", "none", "n/a", "na", "-"}:
                return None
            return s

        return v

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Invalid key '{key}' for SensitiveMeta")

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Invalid key '{key}' for SensitiveMeta")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def safe_update(
        self,
        data: Optional["SensitiveMeta | Mapping[str, Any]"] = None,
        **kwargs: Any,
    ) -> None:
        """
        Safely update fields from:
        - a SensitiveMeta instance
        - a mapping/dict
        - keyword arguments

        Unknown keys and blank values are ignored.
        Existing non-blank values are never downgraded.
        """
        payload: Dict[str, Any] = {}

        if isinstance(data, BaseModel):
            payload.update(data.model_dump())
        elif isinstance(data, Mapping):
            payload.update(dict(data))
        elif data is not None:
            logger.warning(
                "SensitiveMeta.safe_update: unsupported data type %r – ignoring.",
                type(data),
            )
            return

        if kwargs:
            payload.update(kwargs)

        if not payload:
            return

        try:
            validated_updates = SensitiveMeta(**payload)
        except Exception as e:
            logger.error(f"Failed to validate updates in safe_update: {e}")
            return

        for field, new_value in validated_updates.model_dump().items():
            if new_value is not None:
                setattr(self, field, new_value)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensitiveMeta":
        return cls(**(data or {}))
