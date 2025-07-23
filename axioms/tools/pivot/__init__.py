from injector import Module, Binder, singleton

# Re-export from sub-modules.

from axioms.tools.pivot.base import (  # noqa: F401
    PivotSelection,
)

from axioms.tools.pivot.simple import (  # noqa: F401
    RandomPivotSelection,
    FirstPivotSelection,
    LastPivotSelection,
    MiddlePivotSelection,
)


class PivotModule(Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(
            interface=PivotSelection,
            to=RandomPivotSelection,
            scope=singleton,
        )
