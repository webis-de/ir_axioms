from dataclasses import replace
from typing import Callable, Type, TypeVar, cast

from injector import Injector


_T = TypeVar("_T")


# TODO: Can we add the main injector as a default argument?
def lazy_inject(cls: Type[_T], injector: Injector) -> Callable[..., _T]:
    def wrapped(**kwargs) -> _T:
        obj = injector.get(cls)
        return cast(
            _T,
            replace(obj, **kwargs),  # type: ignore
        )

    return wrapped
