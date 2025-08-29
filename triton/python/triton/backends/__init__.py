import importlib
import inspect
import sys
from dataclasses import dataclass
from typing import Type, TypeVar, Union
from types import ModuleType
from .driver import DriverBase
from .compiler import BaseBackend

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

T = TypeVar("T", bound=Union[BaseBackend, DriverBase])


def _find_concrete_subclasses(module: ModuleType, base_class: Type[T]) -> Type[T]:
    ret: list[Type[T]] = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and not inspect.isabstract(attr):
            ret.append(attr)
    if len(ret) == 0:
        raise RuntimeError(f"Found 0 concrete subclasses of {base_class} in {module}: {ret}")
    if len(ret) > 1:
        raise RuntimeError(f"Found >1 concrete subclasses of {base_class} in {module}: {ret}")
    return ret[0]


@dataclass(frozen=True)
class Backend:
    compiler: Type[BaseBackend]
    driver: Type[DriverBase]


def _discover_backends() -> dict[str, Backend]:
    backends = dict()
    from triton.backends import cpu

    backends["cpu"] = Backend(_find_concrete_subclasses(cpu.compiler, BaseBackend),
                              _find_concrete_subclasses(cpu.driver, DriverBase))
    return backends


backends: dict[str, Backend] = _discover_backends()
