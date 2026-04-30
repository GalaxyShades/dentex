"""Command-line entry point for local DENTEX research utilities."""

from __future__ import annotations

import argparse

from core.paths import ensure_project_dirs


def main() -> None:
    parser = argparse.ArgumentParser(prog="dentex")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "init", help="Create local data, model, and results directories"
    )
    args = parser.parse_args()

    if args.command == "init":
        ensure_project_dirs()
        print("Created local DENTEX project directories")


if __name__ == "__main__":
    main()
