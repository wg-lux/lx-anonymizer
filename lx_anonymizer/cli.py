import argparse
import sys
from typing import Protocol, cast


class _PipelineMain(Protocol):
    def __call__(
        self,
        image_path: str,
        east_model_path: str | None,
        device: str,
        validation: bool,
        min_confidence: float,
        width: int,
        height: int,
    ) -> object | None: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lx-anonymizer",
        description="Run the LX Anonymizer image/PDF processing pipeline.",
    )
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to input image or PDF"
    )
    parser.add_argument(
        "-east",
        "--east",
        type=str,
        required=False,
        help="Path to an EAST text detector model",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="olympus_cv_1500",
        help="Device profile name",
    )
    parser.add_argument(
        "-V",
        "--validation",
        action="store_true",
        help="Return validation metadata in addition to the output path",
    )
    parser.add_argument(
        "-c",
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum probability required to inspect a region",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=320,
        help="Resized image width (should be multiple of 32)",
    )
    parser.add_argument(
        "-e",
        "--height",
        type=int,
        default=320,
        help="Resized image height (should be multiple of 32)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from lx_anonymizer.main_with_reassembly import main as raw_pipeline_main
        from lx_anonymizer.setup.custom_logger import configure_global_logger
    except ImportError as exc:
        parser.exit(
            2,
            f"Missing optional dependency for CLI pipeline: {exc}. "
            "Install with `pip install lx-anonymizer[ocr]`.\n",
        )

    pipeline_main = cast(_PipelineMain, raw_pipeline_main)
    configure_global_logger(verbose=cast(bool, args.verbose))
    result = pipeline_main(
        cast(str, args.image),
        cast(str | None, args.east),
        cast(str, args.device),
        cast(bool, args.validation),
        cast(float, args.min_confidence),
        cast(int, args.width),
        cast(int, args.height),
    )
    if result is not None:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
