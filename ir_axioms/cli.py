from typing import Any

from click import Context, Parameter, echo, group, option

from ir_axioms import __version__


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
