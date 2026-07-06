from lx_dtypes.models import (
    SensitiveMeta,
    SensitiveMetaDataDict,
    SensitiveMetaState,
    SensitiveMetaStateDataDict,
)


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
