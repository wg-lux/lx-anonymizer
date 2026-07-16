"""Typed surface used when the optional PyTorch dependency is not installed.

This intentionally models only the small part of PyTorch used by the project.
When PyTorch is installed, the declarations also provide Pyright with a strict
boundary for an upstream package that does not currently expose complete types.
"""

from typing import Literal

class device:
    type: Literal["cpu", "cuda"] | str

    def __init__(self, device: str) -> None: ...
    def __str__(self) -> str: ...

class dtype: ...

float32: dtype

class _Cuda:
    class OutOfMemoryError(RuntimeError): ...

    def is_available(self) -> bool: ...
    def empty_cache(self) -> None: ...
    def synchronize(self) -> None: ...
    def get_device_name(self, device: int | device | None = None) -> str: ...
    def memory_allocated(self, device: int | device | None = None) -> int: ...
    def memory_reserved(self, device: int | device | None = None) -> int: ...

class _Cudnn:
    benchmark: bool

class _Backends:
    cudnn: _Cudnn

class _Random:
    def manual_seed(self, seed: int) -> object: ...

cuda: _Cuda
backends: _Backends
random: _Random
