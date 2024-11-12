from logging import basicConfig, Logger, getLogger

from class_registry import SortedClassRegistry
from importlib_metadata import version

__version__ = version("axioms")

basicConfig()
logger: Logger = getLogger("axioms")

registry = SortedClassRegistry(
    unique=True,
    attr_name="name",
    sort_key="name",
)
