def is_pyterrier_installed() -> bool:
    """Check if PyTerrier is installed."""
    try:
        import pyterrier  # noqa: F401

        return True
    except ImportError:
        return False


def is_pyserini_installed() -> bool:
    """Check if Pyserini is installed."""
    try:
        import pyserini  # noqa: F401

        return True
    except ImportError:
        return False
