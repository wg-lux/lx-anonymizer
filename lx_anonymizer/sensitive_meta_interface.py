from lx_dtypes.models import (
    SensitiveMeta,
    SensitiveMetaDataDict,
    SensitiveMetaState,
    SensitiveMetaStateDataDict,
)
from typing import Any


def sensitive_meta_to_dict(meta: SensitiveMeta) -> dict[str, Any]:
    """Dump SensitiveMeta while preserving inherited dtype fields."""
    payload = meta.model_dump(mode="json")
    for field_name in SensitiveMeta.model_fields:
        if field_name in payload:
            continue
        value = getattr(meta, field_name, None)
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        elif value is not None:
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
