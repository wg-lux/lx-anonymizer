#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "evaluate_ocr_backend_matrix":
        from lx_anonymizer.evaluation.ocr_backend_matrix import main as evaluation_main

        return evaluation_main(args[1:])

    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="Project management commands.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "evaluate_ocr_backend_matrix",
        help="Run OCR backend matrix evaluation against a golden-set manifest.",
    )
    parser.parse_args(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
