from dataclasses import replace
from typing import Callable, Type, TypeVar, cast

from injector import Injector

from ir_axioms.dependency_injection import injector as _default_injector


_T = TypeVar("_T")


def lazy_inject(
    cls: Type[_T],
    injector: Injector = _default_injector,
) -> Callable[..., _T]:
    def wrapped(**kwargs) -> _T:
        obj = injector.get(cls)
        return cast(
            _T,
            replace(obj, **kwargs),  # type: ignore
        )

    return wrapped
