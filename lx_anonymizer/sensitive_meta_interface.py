# lx_anonymizer/frame_cleaner/sensitive_meta_interface.py
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

# from django.template.defaultfilters import first  # <- not used; remove to avoid import cost
from lx_anonymizer.setup.custom_logger import logger


def _is_blank(v: Any) -> bool:
    """True if value is semantically empty/placeholder."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, (list, dict, set, tuple)) and len(v) == 0:
        return True
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return True
        token = s.casefold()
        if token in {"unknown", "undefined", "null", "none", "n/a", "na", "-"}:
            return True
    return False


def _normalize_scalar(v: Any) -> Any:
    """Light normalization: trim strings; pass through others."""
    if isinstance(v, str):
        return v.strip()
    return v


def _merge_values(current: Any, new: Any) -> Any:
    # If new value is valid, take it (Overwrites current!)
    if not _is_blank(new):
        return new
    return current


@dataclass
class SensitiveMeta:
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
        return asdict(self)

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

        # --- Normalize the main `data` arg into a dict ---
        if isinstance(data, SensitiveMeta):
            payload.update(data.to_dict())
        elif isinstance(data, Mapping):
            payload.update(dict(data))
        elif data is not None:
            # Be defensive: don't crash just because a caller passed the wrong type
            logger.warning(
                "SensitiveMeta.safe_update: unsupported data type %r – ignoring.",
                type(data),
            )

        # --- Merge kwargs on top (kwargs win) ---
        if kwargs:
            payload.update(kwargs)

        if not payload:
            return

        allowed = self.__annotations__.keys()
        for k, v in payload.items():
            if k not in allowed:
                continue

            nv = _normalize_scalar(v)
            if _is_blank(nv):
                continue

            cv = getattr(self, k)
            merged = _merge_values(cv, nv)
            if merged is not cv:
                setattr(self, k, merged)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensitiveMeta":
        data = dict(data or {})
        initial = cls()
        initial.safe_update({k: v for k, v in data.items() if k in cls.__annotations__})
        return initial
