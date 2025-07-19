from dataclasses import replace
from typing import Callable, Type, TypeVar, cast

from injector import Injector


_T = TypeVar("_T")


def lazy_inject(cls: Type[_T], injector: Injector) -> Callable[..., _T]:
    def wrapped(**kwargs) -> _T:
        obj = injector.get(cls)
        return cast(
            _T,
            replace(obj, **kwargs),  # type: ignore
        )

    return wrapped
