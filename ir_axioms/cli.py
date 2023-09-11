from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Optional

from click import Context, echo, group, option, Path as PathType, \
    Choice, argument, pass_obj, pass_context
from pandas import read_json, DataFrame
from pyterrier import started, init, __version__ as pyterrier_version
from pyterrier.io import read_results
from typing_extensions import Literal

from ir_axioms import __version__ as ir_axioms_version
from ir_axioms.axiom import to_axiom, Axiom


@dataclass(frozen=True)
class CliOptions:
    terrier_version: Optional[str]
    terrier_helper_version: Optional[str]
    offline: bool


@group(help="Intuitive interface to many IR axioms.")
@option("--terrier-version", type=str, envvar="TERRIER_VERSION")
@option("--terrier-helper-version", type=str, envvar="TERRIER_HELPER_VERSION")
@option("--offline/--online", default=False, envvar="IR_AXIOMS_OFFLINE")
@pass_context
def cli(context: Context, terrier_version: Optional[str],
        terrier_helper_version: Optional[str], offline: bool) -> None:
    context.obj = CliOptions(
        terrier_version=terrier_version,
        terrier_helper_version=terrier_helper_version,
        offline=offline,
    )


@cli.command()
@pass_obj
def version(cli_options: CliOptions) -> None:
    other_versions = [f"PyTerrier {pyterrier_version}"]
    if cli_options.terrier_version is not None:
        other_versions.append(f"Terrier {cli_options.terrier_version}")
    if cli_options.terrier_helper_version is not None:
        other_versions.append(f"Terrier Helper "
                              f"{cli_options.terrier_helper_version}")
    echo(f"{ir_axioms_version} ({', '.join(other_versions)})")


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
@option("--output-minimal", is_flag=True, default=True)
@option("--output-compress", type=Choice(["gzip", None]), default="gzip")
@argument("axiom", nargs=-1, required=True)
@pass_obj
def preferences(cli_options: CliOptions, run_path: Path,
                run_format: Literal["trec", "letor", "jsonl"],
                index_path: Path, output_path: Path, axiom: Sequence[str],
                output_minimal: bool,  output_compress: Literal["gzip", None]
                ) -> None:
    axiom_names = axiom

    if not started():
        suffix = ""
        if cli_options.offline is not None:
            suffix += " (offline)"
        echo(f"Initialize PyTerrier{suffix}.")
        init(
            version=cli_options.terrier_version,
            helper_version=cli_options.terrier_helper_version,
            no_download=cli_options.offline,
            tqdm="auto",
        )
        import ir_axioms.utils.nltk
        ir_axioms.utils.nltk.offline = cli_options.offline

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

    output_path.mkdir(exist_ok=True)
    output_path = output_path / ('preferences.jsonl' + ('.gz' if output_compress == 'gzip' else ''))
    echo(f"Save axiomatic preferences to: {output_path}")
    shared_columns = set(all_preferences.columns) & original_columns
    non_shared_columns = original_columns - shared_columns
    select_columns = (shared_columns |
                      {f"{col}_a" for col in non_shared_columns} |
                      {f"{col}_b" for col in non_shared_columns})

    if output_minimal:
        for i in ['query', 'text_a', 'score_a', 'text_b', 'score_b']:
            del all_preferences[i]

    all_preferences.to_json(output_path, orient="records", lines=True)

    echo("Done.")
