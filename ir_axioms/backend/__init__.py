from typing import ContextManager


class PyseriniBackend(ContextManager):
    def __init__(self):
        try:
            import pyserini  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "The Pyserini backend requires that 'pyserini' is installed."
            ) from error

    def __exit__(self, type, value, traceback):
        return None
