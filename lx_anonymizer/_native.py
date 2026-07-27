from collections.abc import Iterable
from importlib import import_module
from types import ModuleType
from typing import cast

native: ModuleType | None

try:
    native = import_module("lx_anonymizer._lx_anonymizer_native")
except ImportError:
    try:
        native = import_module("_lx_anonymizer_native")
    except ImportError:
        native = None


def available_native_capabilities() -> frozenset[str]:
    if native is None:
        return frozenset()
    capability_function = getattr(native, "native_capabilities", None)
    if not callable(capability_function):
        return frozenset()
    raw_capabilities = capability_function()
    if not isinstance(raw_capabilities, Iterable) or isinstance(
        raw_capabilities, (str, bytes)
    ):
        raise TypeError("lx-anonymizer native capabilities are malformed")
    capabilities: set[str] = set()
    for capability in cast(Iterable[object], raw_capabilities):
        if not isinstance(capability, str) or not capability.strip():
            raise RuntimeError("lx-anonymizer native capability names are malformed")
        capabilities.add(capability.strip())
    return frozenset(capabilities)


def require_native_capabilities(required: Iterable[str]) -> None:
    required_capabilities = frozenset(item.strip() for item in required if item.strip())
    missing = required_capabilities - available_native_capabilities()
    if missing:
        raise RuntimeError(
            "lx-anonymizer native backend is missing required capabilities: "
            + ", ".join(sorted(missing))
        )
