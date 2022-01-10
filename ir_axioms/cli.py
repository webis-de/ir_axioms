from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

from ir_axioms import registry
from ir_axioms.app import save_rerank_run, save_rerank_ir_datasets_run
from ir_axioms.axiom import Axiom


def _prepare_parser(parser: ArgumentParser) -> ArgumentParser:
    parsers = parser.add_subparsers(title="subcommands", dest="command")
    _prepare_parser_rerank_run(parsers.add_parser(
        "rerank-run",
        aliases=["rerank"]
    ))
    _prepare_parser_rerank_ir_datasets_run(parsers.add_parser(
        "rerank-ir-datasets-run",
        aliases=["rerank-ird"]
    ))
    return parser


def _prepare_parser_rerank_run(parser: ArgumentParser) -> ArgumentParser:
    import ir_axioms.axiom.all  # noqa (Import all axiom implementations.)
    parser.add_argument(
        "--axiom", "-a",
        dest="axioms",
        type=str,
        choices=set(registry.keys()),
        default=[],
        action="append",
        required=True,
    )
    parser.add_argument(
        dest="run",
        type=Path,
    )
    parser.add_argument(
        dest="topics",
        type=Path,
    )
    parser.add_argument(
        dest="index",
        type=Path,
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default="ir_axioms",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        type=Path,
        default=None
    )
    parser.add_argument(
        dest="out",
        type=Path,
    )
    return parser


def _prepare_parser_rerank_ir_datasets_run(
        parser: ArgumentParser
) -> ArgumentParser:
    import ir_axioms.axiom.all  # noqa (Import all axiom implementations.)
    parser.add_argument(
        "--axiom", "-a",
        dest="axioms",
        type=str,
        choices=set(registry.keys()),
        default=[],
        action="append",
        required=True,
    )
    parser.add_argument(
        dest="run",
        type=Path,
    )
    parser.add_argument(
        dest="dataset",
        type=str,
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default="ir_axioms",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        type=Path,
        default=None
    )
    parser.add_argument(
        dest="out",
        type=Path,
    )
    return parser


def _parse_axiom(axiom: str) -> Axiom:
    if axiom in registry.keys():
        return registry[axiom]
    else:
        raise Exception(f"Unknown axiom: {axiom}")


def _parse_axioms(
        axioms: List[str]
) -> List[Axiom]:
    if len(axioms) < 1:
        raise Exception("Must specify at least one axiom.")
    return [_parse_axiom(axiom) for axiom in axioms]


def main():
    parser: ArgumentParser = ArgumentParser()
    _prepare_parser(parser)
    args: Namespace = parser.parse_args()
    if args.command in ["rerank-run", "rerank"]:
        axiom: List[Axiom] = _parse_axioms(args.axioms)
        run: Path = args.run
        reranked_run_path: Path = args.out
        topics: Path = args.topics
        context: Path = args.index
        tag: str = args.tag
        cache_dir: Path = args.cache
        save_rerank_run(
            axiom,
            run,
            reranked_run_path,
            topics,
            context,
            tag,
            cache_dir
        )
    elif args.command in ["rerank-ir-datasets-run", "rerank-ird"]:
        axiom: List[Axiom] = _parse_axioms(args.axioms)
        run: Path = args.run
        reranked_run_path: Path = args.out
        dataset: str = args.dataset
        tag: str = args.tag
        cache_dir: Path = args.cache
        save_rerank_ir_datasets_run(
            axiom,
            run,
            reranked_run_path,
            dataset,
            tag,
            cache_dir
        )
    else:
        parser.print_help()
