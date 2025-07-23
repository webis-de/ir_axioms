from typing import Annotated

from cyclopts import App, Parameter
from dotenv import load_dotenv, find_dotenv

app = App()


# Load .env file before running any command.
@app.meta.default
def launcher(*tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)]):
    if find_dotenv():
        load_dotenv()
    command, bound, _ = app.parse_args(tokens=tokens)
    return command(*bound.args, **bound.kwargs)


@app.command()
def dummy() -> None:
    return
