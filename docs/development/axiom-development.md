# Developing Axioms

<p class="subtitle">Turn an intuition into a new axiom</p>

Follow these steps to add new axioms to this project:

1. Find a desirable property that a good document ranking or generated text response should fulfill.
   (At this stage, a rough intuition like in the [list of missing axioms](../axioms/generation.md#missing-axioms-and-axiom-ideas) often suffices.)
2. Formalize your understanding into a _pairwise_ preference between two documents or responses.
3. Determine dependencies (and optionally, look for [existing tool implementations](https://github.com/webis-de/ir_axioms/tree/main/ir_axioms/tools)) for computing the preference.
   The available tools and their default implementations are listed in the [core concepts](../getting-started/concepts.md#tools-and-dependency-injection).
4. Extend the generic `Axiom` class, e.g., for generated responses:

    ```python
    from dataclasses import dataclass
    from math import isclose

    from injector import inject, NoInject

    from ir_axioms.axiom import Axiom
    from ir_axioms.axiom.utils import strictly_less
    from ir_axioms.model import GenerationRequest, GenerationResponse, Preference
    from ir_axioms.tools import TextContents


    @inject
    @dataclass(frozen=True, kw_only=True)
    class YourAxiom(Axiom[GenerationRequest, GenerationResponse]):
        """
        Your intuition.
        """

        # (optional) Specify dependencies on tools.
        text_contents: TextContents[GenerationResponse]

        # Specify further parameters to customize your axiom, e.g., a margin fraction for float comparisons.
        margin_fraction: NoInject[float] = 0.1

        def preference(
            self,
            input: GenerationRequest,
            output1: GenerationResponse,
            output2: GenerationResponse,
        ) -> Preference:
            some_proxy1 = do_something(self.text_contents.contents(output1))
            some_proxy2 = do_something(self.text_contents.contents(output2))

            # Catch unclear cases with margin fraction.
            if isclose(some_proxy1, some_proxy2, rel_tol=self.margin_fraction):
                return 0

            # Compare and return the axiom's preference.
            return strictly_less(some_proxy1, some_proxy2)
    ```

5. (optional) Override the batched preference computation to speed up experiments
   (using `Sequence` from `collections.abc`, `array` and `float_` from `numpy`, and `PreferenceMatrix` from `ir_axioms.model`):

    ```python
    def preferences(
        self,
        input: GenerationRequest,
        outputs: Sequence[GenerationResponse],
    ) -> PreferenceMatrix:
        # Compute only once for each output.
        some_proxy = [
            do_something(self.text_contents.contents(output))
            for output in outputs
        ]
        # Return a NumPy float array as the preference matrix of the axiom.
        return array([
            (
                strictly_less(some_proxy1, some_proxy2)
                if not isclose(some_proxy1, some_proxy2, rel_tol=self.margin_fraction)
                else 0
            )
            for some_proxy1 in some_proxy
            for some_proxy2 in some_proxy
        ], dtype=float_).reshape((len(outputs), len(outputs)))
    ```

6. Make the axiom available under a short name:

    ```python
    from typing import Final
    from ir_axioms.utils.lazy import lazy_inject

    YOU: Final = lazy_inject(YourAxiom)
    ```

7. Check out the [RAG evaluation walkthrough](../walkthroughs/rag-evaluation.md) (based on the [ICTIR 2025 notebook](https://github.com/webis-de/ir_axioms/blob/main/experiments/ictir2025.ipynb)) to evaluate the axiom.

!!! tip

    For further examples, refer to the [existing axiom implementations](https://github.com/webis-de/ir_axioms/tree/main/ir_axioms/axiom).

!!! warning "Preconditions and overridden methods"

    The `PreconditionMixin` applies its precondition by wrapping `preference()` and `preferences()`.
    If your axiom extends `PreconditionMixin`, implement the axiom's logic in a base class (as done for `LenAndAxiom` extending `AndAxiom`), so that the mixin's methods are not shadowed by your own `preference()` and `preferences()` methods.

## Tool Bindings

The available tools and their default bindings are described in the [core concepts](../getting-started/concepts.md#tools-and-dependency-injection).
To use a different tool implementation for all axioms, bind it in the default injector.
If axioms were already instantiated before, also reset the binding scopes, so that singleton tools are re-created:

```python
from injector import singleton
from ir_axioms.dependency_injection import injector
from ir_axioms.tools import AspectExtraction, SpacyEntitiesAspectExtraction
from ir_axioms.utils.injection import reset_binding_scopes

injector.binder.bind(
    interface=AspectExtraction,
    to=SpacyEntitiesAspectExtraction(),
    scope=singleton,
)
reset_binding_scopes(injector)
```

Axioms instantiated afterwards (e.g., `COV1()`) then use the new tool.
