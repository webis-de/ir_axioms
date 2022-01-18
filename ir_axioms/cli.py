from argparse import ArgumentParser, Namespace

from ir_axioms import __version__


def _prepare_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-v", "--version",
        dest="print_version",
        default=False,
        action="store_true",
        help="show version number and exit"
    )
    return parser


def main():
    parser: ArgumentParser = ArgumentParser(
        "ir_axioms",
        description="Intuitive interface to many IR axioms.",
    )
    _prepare_parser(parser)
    args: Namespace = parser.parse_args()
    print_version: bool = args.print_version

    if print_version:
        print(__version__)
        return

    parser.print_help()
