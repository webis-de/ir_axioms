from socket import create_connection

from nltk.downloader import Downloader

from axioms.logging import logger

_downloader = Downloader()


def _is_connected() -> bool:
    try:
        create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False


def download_nltk_dependencies(*dependencies: str) -> None:
    if not _is_connected():
        return
    for dependency in dependencies:
        if not _downloader.is_installed(dependency):
            logger.info(f"Downloading NLTK dependency {dependency}.")
            _downloader.download(dependency)
