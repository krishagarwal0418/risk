"""Launch the FastAPI service with uvicorn."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run("safety_classifier.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
