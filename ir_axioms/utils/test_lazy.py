from dataclasses import dataclass
from math import isclose

from injector import Injector, inject
from pytest import raises

from ir_axioms.utils.lazy import lazy_inject


def test_lazy() -> None:
    injector = Injector()
    injector.binder.bind(int, 2)
    injector.binder.bind(str, "two")
    injector.binder.bind(float, 2.0)

    @inject
    @dataclass(frozen=True, kw_only=True)
    class A:
        foo: int
        bar: str

    a1 = A(foo=1, bar="one")
    assert a1.foo == 1
    assert a1.bar == "one"

    a2 = injector.get(A)
    assert a2.foo == 2
    assert a2.bar == "two"

    with raises(TypeError):
        A(foo=21)  # type: ignore

    @inject
    @dataclass(frozen=True, kw_only=True)
    class B(A):
        baz: float

    b1 = B(foo=1, bar="one", baz=1.0)
    assert b1.foo == 1
    assert b1.bar == "one"
    assert isclose(b1.baz, 1.0)

    b2 = injector.get(B)
    assert b2.foo == 2
    assert b2.bar == "two"
    assert isclose(b2.baz, 2.0)

    C = lazy_inject(B, injector=injector)

    c1 = C()
    assert c1.foo == 2
    assert c1.bar == "two"
    assert isclose(c1.baz, 2.0)

    c2 = C(foo=3, bar="three", baz=3.0)
    assert c2.foo == 3
    assert c2.bar == "three"
    assert isclose(c2.baz, 3.0)
