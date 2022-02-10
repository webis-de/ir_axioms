__version__ = "0.1.0"

from logging import basicConfig, Logger, getLogger

from ir_axioms.axiom.registry import registry as _registry

basicConfig()
logger: Logger = getLogger(__name__)

registry = _registry
