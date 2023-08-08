from pathlib import Path
from typing import Any, Dict, Sequence
from typing_extensions import Literal

from click import Context, Parameter, echo, group, option, Path as PathType, \
    Choice, argument
from pandas import read_json, DataFrame
from pyterrier import started, init
from pyterrier.io import read_results

from ir_axioms import __version__
from ir_axioms.axiom import to_axiom, Axiom


def print_version(
        context: Context,
        _parameter: Parameter,
        value: Any,
) -> None:
    if not value or context.resilient_parsing:
        return
    echo(__version__)
    context.exit()


@group(help="Intuitive interface to many IR axioms.")
@option("-V", "--version", is_flag=True, callback=print_version,
        expose_value=False, is_eager=True)
def cli() -> None:
    pass


@cli.command()
@option("-r", "--run-file", "run_path",
        type=PathType(path_type=Path, exists=True, file_okay=True,
                      dir_okay=False, readable=True, writable=False,
                      resolve_path=True, allow_dash=False),
        required=True)
@option("--run-format", type=Choice(["trec", "letor", "jsonl"]),
        default="trec")
@option("-i", "--index-dir", "index_path",
        type=PathType(path_type=Path, exists=True, file_okay=False,
                      dir_okay=True, readable=True, writable=False,
                      resolve_path=True, allow_dash=False),
        required=True)
@option("-o", "--output-dir", "output_path",
        type=PathType(path_type=Path, exists=False, file_okay=False,
                      dir_okay=True, readable=False, writable=True,
                      resolve_path=True, allow_dash=False),
        required=True)
@argument("axiom", nargs=-1, required=True)
def preferences(run_path: Path, run_format: Literal["trec", "letor", "jsonl"],
                index_path: Path, output_path: Path, axiom: Sequence[str]
                ) -> None:
    axiom_names = axiom

    if not started():
        echo("Initialize PyTerrier.")
        init()

    echo(f"Read run from: {run_path}")
    run: DataFrame
    if run_format == "trec" or run_format == "letor":
        run = read_results(str(run_path), format=run_format)
    elif run_format == "jsonl":
        run = read_json(run_path, lines=True)
        for col in ["original_document", "original_query"]:
            if col in run.columns:
                del run[col]
    else:
        raise ValueError(f"Unknown run format: {run_format}")
    original_columns = set(run.columns)

    echo(f"Load axioms: {', '.join(axiom_names)}")
    axioms: Dict[str, Axiom] = {name: to_axiom(name) for name in axiom_names}

    echo(f"Create axiomatic preference transformer using index "
         f"from: {index_path}")
    from ir_axioms.backend.pyterrier.transformers import AxiomaticPreferences
    pipeline = AxiomaticPreferences(
        axioms=list(axioms.values()),
        index=index_path,
        verbose=True,
    )

    echo("Compute axiomatic preferences.")
    all_preferences = pipeline.transform(run)

    echo(f"Save axiomatic preferences to: {output_path}")
    output_path.mkdir(exist_ok=True)
    shared_columns = set(all_preferences.columns) & original_columns
    non_shared_columns = original_columns - shared_columns
    select_columns = (shared_columns |
                      {f"{col}_a" for col in non_shared_columns} |
                      {f"{col}_b" for col in non_shared_columns})
    for name, axiom in axioms.items():
        axiom_column = f"{axiom}_preference"
        axiom_columns = select_columns | {axiom_column}
        preferences: DataFrame = all_preferences[list(axiom_columns)].copy()
        preferences.rename(columns={axiom_column: "preference"}, inplace=True)
        preferences.to_json(output_path / f"{name}.jsonl", orient="records",
                            lines=True)

    echo("Done.")
