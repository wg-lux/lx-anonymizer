from collections.abc import Mapping
from datetime import date
from typing import cast

from lx_dtypes.models.meta.SensitiveMeta import SensitiveMeta as DTypeSensitiveMeta
from lx_dtypes.models.meta.SensitiveMeta import (
    SensitiveMetaDataDict,
    SensitiveMetaState,
    SensitiveMetaStateDataDict,
)
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class SensitiveMeta(DTypeSensitiveMeta):
    """Stable, mixed-input boundary for sensitive metadata.

    Upstream extractors intentionally pass mappings containing non-sensitive
    fields.  This compatibility layer projects those mappings onto the shared
    dtype, repairs the common OCR error where birth and examination dates are
    reversed, and keeps fill-only updates transactional.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_input_payload(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data

        normalized: dict[str, object] = {}
        typed_data = cast(Mapping[object, object], data)
        for raw_key, raw_value in typed_data.items():
            key = cls._LEGACY_FIELD_ALIASES.get(str(raw_key), str(raw_key))
            if key not in cls.model_fields:
                continue
            if key in normalized and cls._is_nonblank(normalized[key]):
                continue
            normalized[key] = cls._normalize_value(raw_value, key)

        dob = cls._parsed_date(normalized.get("dob"))
        examination_date = cls._parsed_date(normalized.get("examination_date"))
        if dob is not None and examination_date is not None and examination_date < dob:
            normalized["dob"], normalized["examination_date"] = (
                examination_date,
                dob,
            )
        return normalized

    @classmethod
    def _parsed_date(cls, value: object) -> date | None:
        """Narrow normalized date input without accepting datetimes."""
        return cls._parse_date_like(value)

    def safe_update(
        self,
        data: object = None,
        **kwargs: object,
    ) -> None:
        """Fill blank fields from a mixed payload as one atomic update."""
        payload: dict[str, object] = {}
        if isinstance(data, BaseModel):
            payload.update(data.model_dump())
        elif isinstance(data, Mapping):
            for key, value in cast(Mapping[object, object], data).items():
                payload[str(key)] = value
        elif data is not None:
            return

        payload.update(kwargs)
        if not payload:
            return

        model_type = type(self)
        try:
            validated_updates = model_type.from_mixed_mapping(payload)
        except ValidationError:
            return

        excluded_fields = {"created_at", "sensitive_meta_state", "uuid"}
        fill_updates = {
            field: getattr(validated_updates, field)
            for field in model_type.model_fields
            if field not in excluded_fields
            and self._is_nonblank(getattr(validated_updates, field))
            and not self._is_nonblank(getattr(self, field))
        }
        if not fill_updates:
            return

        try:
            merged = model_type.model_validate(self.model_dump() | fill_updates)
        except ValidationError:
            return

        # Preserve validated nested model instances; model_dump() would flatten
        # SensitiveMetaState into an untyped dictionary here.
        for field in model_type.model_fields:
            object.__setattr__(self, field, getattr(merged, field))
        self.__pydantic_fields_set__.update(fill_updates)


def sensitive_meta_to_dict(meta: SensitiveMeta) -> dict[str, object]:
    """Dump SensitiveMeta while preserving inherited dtype fields."""
    payload = meta.model_dump(mode="json")
    for field_name in SensitiveMeta.model_fields:
        if field_name in payload:
            continue
        value = getattr(meta, field_name, None)
        if value is not None:
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            else:
                value = str(value)
        payload[field_name] = value
    return payload


__all__ = [
    "SensitiveMeta",
    "SensitiveMetaDataDict",
    "SensitiveMetaState",
    "SensitiveMetaStateDataDict",
    "sensitive_meta_to_dict",
]
