from typing import ContextManager


def is_pyserini_installed() -> bool:
    try:
        import pyserini  # noqa: F401
        return True
    except ImportError:
        return False


def is_pyterrier_installed() -> bool:
    try:
        import pyterrier  # noqa: F401
        return True
    except ImportError:
        return False


class PyseriniBackendContext(ContextManager):
    def __init__(self):
        try:
            import pyserini  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "The Pyserini backend requires that 'pyserini' is installed."
            ) from error

    def __exit__(self, *args):
        return None


class PyTerrierBackendContext(ContextManager):
    def __init__(self):
        try:
            import pyterrier  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "The PyTerrier backend requires that 'python-terrier' "
                "is installed."
            ) from error

        from pyterrier import started, init
        if not started():
            init()
            # print("Not started")

    def __exit__(self, *args):
        return None
