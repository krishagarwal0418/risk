"""Download and adapt all configured datasets into data/raw/."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from safety_classifier.data.dataset_downloader import download_all, print_report


def main() -> None:
    metadata = download_all()
    print_report(metadata)


if __name__ == "__main__":
    main()
