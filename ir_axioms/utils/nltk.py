from logging import info

from nltk.downloader import Downloader


def download_nltk_dependencies(*dependencies: str):
    downloader = Downloader()
    for dependency in dependencies:
        if not downloader.is_installed(dependency):
            info(f"Downloading NLTK dependency {dependency}.")
            downloader.download(dependency)
