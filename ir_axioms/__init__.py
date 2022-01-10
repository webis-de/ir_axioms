from logging import basicConfig, Logger, getLogger
from typing import List

__version__ = "0.1.0"

basicConfig()
logger: Logger = getLogger(__name__)

def main(args: List[str]):
    raise NotImplementedError()


def cli():
    import sys
    main(sys.argv)
