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


def is_keybert_installed() -> bool:
    """Check if KeyBERT is installed."""
    try:
        import keybert  # noqa: F401

        return True
    except ImportError:
        return False
        return False


def is_sentence_transformers_installed() -> bool:
    """Check if Sentence Transformers is installed."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False
