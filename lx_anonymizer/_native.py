from importlib import import_module
from types import ModuleType


native: ModuleType | None

try:
    native = import_module("lx_anonymizer._lx_anonymizer_native")
except ImportError:
    try:
        native = import_module("_lx_anonymizer_native")
    except ImportError:
        native = None
