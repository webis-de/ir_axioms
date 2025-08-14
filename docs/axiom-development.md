# Axiom Development

Follow these steps to adding new axioms to this project:

1. Find a desirable property that a good document ranking or generated text response should fulfill. (At this stage, a rough intuition like in the [list of missing axioms](./axioms.md#missing-axioms-and-axiom-ideas) often suffices.)
1. Formalize your understanding into a _pairwise_ preference between two documents or responses.
1. Determine dependencies (and optionally, look for [existing tool implementations](axioms/tools/__init__.py)) for computing the preference.
1. Extend the generic `Axiom` class, e.g., for generated responses:

    ```python
    @inject
    @dataclass(frozen=True, kw_only=True)
    class YourAxiom(Axiom[GenerationInput, GenerationOutput]):
        """
        Your intuition.
        """

        # (optional) Specify dependencies on tools.
        text_contents: TextContents[GenerationOutput]

        # Specify further parameters to customize your axiom, e.g., a margin fraction for float comparisons.
        margin_fraction: NoInject[float] = 0.1

        def preference(
            self,
            input: GenerationInput,
            output1: GenerationOutput,
            output2: GenerationOutput,
        ) -> Preference:
            some_proxy1 = do_something(self.text_contents.contents(output1))
            some_proxy2 = do_something(self.text_contents.contents(output2))

            # Catch unclear cases with margin fraction
            if isclose(some_proxy1, some_proxy2, rel_tol=self.margin_fraction):
                return 0

            # Compare and return the axiom's preference.
            return strictly_less(some_proxy1, some_proxy2)
    ```

1. (optional) Override the batched preference computation to speed up experiments:

    ```python
    def preferences(
        self,
        input: GenerationInput,
        outputs: Sequence[GenerationOutput],
    ) -> PreferenceMatrix:
        # Compute only once for each output.
        some_proxy = [
            do_something(self.text_contents.contents(output))
            for output in outputs
        ]
        # Return a Numpy float array as the preference matrix of the axiom.
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

1. Make the axiom available under a short name: `YOU = lazy_inject(YourAxiom)`

1. Check out the [example evaluation notebook](../experiments/ictir.ipynb) to evaluate the axiom.

Note: For further examples, refer to the [existing axiom implementations](../axioms/axiom/__init__.py).
