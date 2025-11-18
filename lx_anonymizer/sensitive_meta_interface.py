# lx_anonymizer/frame_cleaner/sensitive_meta_interface.py
from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict, Mapping, Iterable
import math

# from django.template.defaultfilters import first  # <- not used; remove to avoid import cost
from .text_anonymizer import anonymize_text


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
    """
    Merge 'new' into 'current' safely:
    - If current is blank and new is not ⇒ take new
    - If both dicts ⇒ shallow-safe merge (without downgrades)
    - If both lists/tuples ⇒ stable union preserving order
    - Otherwise: keep current unless current is blank
    """
    if _is_blank(current) and not _is_blank(new):
        return new

    if isinstance(current, dict) and isinstance(new, dict):
        out = dict(current)
        for k, nv in new.items():
            cv = out.get(k)
            if _is_blank(nv):
                continue
            out[k] = nv if _is_blank(cv) else cv
        return out

    if isinstance(current, (list, tuple)) and isinstance(new, (list, tuple)):
        seen = set()
        out_list = []
        for x in list(current) + list(new):
            key = (type(x).__name__, str(x))
            if key not in seen and not _is_blank(x):
                seen.add(key)
                out_list.append(x)
        return type(current)(out_list) if isinstance(current, tuple) else out_list

    # scalar vs scalar: only replace if current is blank
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

    def safe_update(self, data: Optional[Mapping[str, Any]] = None, **kwargs) -> None:
        """
        Safely update fields from a mapping or keyword arguments:
        - Ignore unknown keys
        - Ignore blank values (None, '', {}, [], 'Unknown', 'undefined', NaN, etc.)
        - Never overwrite a non-blank existing value with a blank or conflicting value
        - Merge dicts/lists without duplicates, preserving existing content
        """
        payload: Dict[str, Any] = {}

        if data:
            payload.update(dict(data))
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
            # Only set when there is a meaningful improvement/change
            if merged is not cv:
                setattr(self, k, merged)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensitiveMeta":
        """
        Construct from dict safely. Also auto-create anonymized_text if missing
        and (first_name, last_name, text) are present (legacy compatibility).
        """
        data = dict(data or {})
        if (
            not data.get("anonymized_text")
            and data.get("patient_first_name")
            and data.get("patient_last_name")
            and data.get("text")
        ):
            data["anonymized_text"] = anonymize_text(
                report_meta=data,
                text=data["text"],
                first_names=[data["patient_first_name"]],
                last_names=[data["patient_last_name"]],
            )
        # Keep only declared fields; then use safe_update to avoid downgrades
        initial = cls()
        initial.safe_update({k: v for k, v in data.items() if k in cls.__annotations__})
        return initial
