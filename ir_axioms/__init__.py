from logging import basicConfig, Logger, getLogger

from class_registry import SortedClassRegistry

__version__ = "0.1.1"

basicConfig()
logger: Logger = getLogger(__name__)

registry = SortedClassRegistry(
    unique=True,
    attr_name="name",
    sort_key="name",
)
