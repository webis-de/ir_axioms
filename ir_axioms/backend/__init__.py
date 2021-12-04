from contextlib import contextmanager


@contextmanager
def require_pyserini_backend():
    try:
        import pyserini  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "The Pyserini backend requires that 'pyserini' is installed."
        ) from error
