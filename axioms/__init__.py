from logging import basicConfig, Logger, getLogger

from importlib_metadata import version

__version__ = version("axioms")

basicConfig()
logger: Logger = getLogger("axioms")
