from logging import info

from nltk.downloader import Downloader

_downloader = Downloader()


def download_nltk_dependencies(*dependencies: str) -> None:
    for dependency in dependencies:
        if not _downloader.is_installed(dependency):
            info(f"Downloading NLTK dependency {dependency}.")
            _downloader.download(dependency)
