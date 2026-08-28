from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID, uuid4


class _ImagePipelineMain(Protocol):
    def __call__(
        self,
        image_path: str,
        east_model_path: Path | None,
        device: str,
        validation: bool,
        min_confidence: float,
        width: int,
        height: int,
    ) -> object | None: ...


class _CommandMain(Protocol):
    def __call__(self, argv: Sequence[str] | None = None) -> int: ...


@dataclass(frozen=True)
class _DelegatedCommand:
    name: str
    description: str
    module: str


_DELEGATED_COMMANDS = (
    _DelegatedCommand(
        "export-dicom",
        "Create an anonymized DICOM tree and validation artifacts.",
        "lx_anonymizer.dicom_anonymization",
    ),
    _DelegatedCommand(
        "evaluate-midi-b",
        "Evaluate PHI-region detection against MIDI-B answer boxes.",
        "lx_anonymizer.evaluation.midi_b",
    ),
    _DelegatedCommand(
        "generate-phi-data",
        "Generate synthetic PHI-region training frames.",
        "lx_anonymizer.training.synthetic_phi_frames",
    ),
    _DelegatedCommand(
        "generate-endoscopy-stickers",
        "Generate synthetic endoscopy sticker training data.",
        "lx_anonymizer.training.synthetic_endoscopy_stickers",
    ),
    _DelegatedCommand(
        "generate-midi-b-phi-data",
        "Convert MIDI-B annotations into a PHI-region dataset.",
        "lx_anonymizer.training.midi_b_phi_dataset",
    ),
    _DelegatedCommand(
        "generate-radphi-data",
        "Convert a Rad-PHI dataset into the training layout.",
        "lx_anonymizer.training.radphi_dataset",
    ),
    _DelegatedCommand(
        "train-phi",
        "Train and export the PHI-region detector.",
        "lx_anonymizer.text_detection.phi_region_detector_training",
    ),
)
_DELEGATED_COMMAND_BY_NAME = {command.name: command for command in _DELEGATED_COMMANDS}


def build_parser() -> argparse.ArgumentParser:
    command_lines = [
        "  image                         Anonymize one image or PDF.",
        "  report                        Anonymize one immutable PDF report.",
    ]
    command_lines.extend(
        f"  {command.name:<29} {command.description}" for command in _DELEGATED_COMMANDS
    )
    return argparse.ArgumentParser(
        prog="lx-anonymizer",
        usage="%(prog)s COMMAND [ARGS]",
        description="Anonymize medical images and reports, or run project utilities.",
        epilog=(
            "Commands:\n"
            + "\n".join(command_lines)
            + "\n\nRun `lx-anonymizer COMMAND --help` for command-specific options.\n"
            "The legacy `lx-anonymizer -i INPUT` form remains supported."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def build_image_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lx-anonymizer image",
        description="Run the standalone image/PDF pixel anonymization pipeline.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input image or PDF")
    parser.add_argument(
        "-i",
        "--image",
        dest="legacy_input",
        type=Path,
        help="Input image or PDF (legacy spelling)",
    )
    parser.add_argument(
        "--east",
        "-east",
        type=Path,
        help="Path to an EAST text detector model",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="olympus_cv_1500",
        help="Device profile name",
    )
    parser.add_argument(
        "-V",
        "--validation",
        action="store_true",
        help="Print validation metadata in addition to the output path",
    )
    parser.add_argument(
        "-c",
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum probability required to inspect a region (default: 0.5)",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=320,
        help="Detector input width, as a multiple of 32 (default: 320)",
    )
    parser.add_argument(
        "-e",
        "--height",
        type=int,
        default=320,
        help="Detector input height, as a multiple of 32 (default: 320)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser


def build_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lx-anonymizer report",
        description=(
            "Anonymize one immutable PDF into an unpublished, attempt-owned candidate."
        ),
    )
    parser.add_argument("source", type=Path, help="Immutable source PDF")
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        required=True,
        help="Attempt-owned output directory",
    )
    parser.add_argument(
        "--attempt-id",
        type=UUID,
        default=None,
        help="Attempt UUID (default: generate a new UUID)",
    )
    parser.add_argument("--locale", help="Replacement-data locale, for example de_DE")
    parser.add_argument(
        "--ensemble", action="store_true", help="Enable ensemble OCR fallback"
    )
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--llm", dest="use_llm", action="store_true", help="Enable LLM extraction"
    )
    llm_group.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        help="Disable LLM extraction",
    )
    parser.set_defaults(use_llm=None)
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Disable verbose pipeline output"
    )
    return parser


def _validated_image_input(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Path:
    positional = cast(Path | None, args.input)
    legacy = cast(Path | None, args.legacy_input)
    if positional is not None and legacy is not None:
        parser.error("provide INPUT or --image, not both")
    source = positional or legacy
    if source is None:
        parser.error("an input image or PDF is required")
    return source


def _run_image(argv: Sequence[str]) -> int:
    parser = build_image_parser()
    args = parser.parse_args(argv)
    source = _validated_image_input(parser, args)

    try:
        from lx_anonymizer.main_with_reassembly import main as raw_pipeline_main
        from lx_anonymizer.setup.custom_logger import configure_global_logger
    except ImportError as exc:
        parser.exit(2, f"The image pipeline installation is incomplete: {exc}.\n")

    pipeline_main = cast(_ImagePipelineMain, raw_pipeline_main)
    configure_global_logger(verbose=cast(bool, args.verbose))
    result = pipeline_main(
        str(source),
        cast(Path | None, args.east),
        cast(str, args.device),
        cast(bool, args.validation),
        cast(float, args.min_confidence),
        cast(int, args.width),
        cast(int, args.height),
    )
    if result is not None:
        print(result)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_report_source(
    parser: argparse.ArgumentParser, source: Path
) -> tuple[Path, int, str]:
    if source.is_symlink():
        parser.error(f"source must not be a symbolic link: {source}")
    try:
        source_stat = source.stat()
    except OSError as exc:
        parser.error(f"source cannot be inspected: {source}: {exc}")
    if not stat.S_ISREG(source_stat.st_mode):
        parser.error(f"source must be a regular file: {source}")
    if source_stat.st_size <= 0:
        parser.error(f"source must not be empty: {source}")
    if source.suffix.casefold() != ".pdf":
        parser.error(f"report source must be a PDF: {source}")
    resolved = source.resolve()
    return resolved, source_stat.st_size, _sha256(resolved)


def _run_report(argv: Sequence[str]) -> int:
    parser = build_report_parser()
    args = parser.parse_args(argv)
    source, source_size, source_sha256 = _validated_report_source(
        parser, cast(Path, args.source)
    )
    output_directory = cast(Path, args.output_directory).expanduser()
    if output_directory.is_symlink():
        parser.error(
            f"output directory must not be a symbolic link: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory = output_directory.resolve()

    from lx_anonymizer.report_contracts import (
        ReportAnonymizationOptions,
        ReportAnonymizationRequest,
    )
    from lx_anonymizer.report_reader import ReportReader

    request = ReportAnonymizationRequest(
        attempt_id=cast(UUID | None, args.attempt_id) or uuid4(),
        source_path=source,
        source_sha256=source_sha256,
        source_size_bytes=source_size,
        output_directory=output_directory,
        options=ReportAnonymizationOptions(
            use_ensemble=cast(bool, args.ensemble),
            verbose=not cast(bool, args.quiet),
            use_llm=cast(bool | None, args.use_llm),
        ),
    )
    result = ReportReader(locale=cast(str | None, args.locale)).process_report(request)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
    return 0


def _run_delegated(command: _DelegatedCommand, argv: Sequence[str]) -> int:
    module = importlib.import_module(command.module)
    command_main = cast(_CommandMain, module.main)
    return command_main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        build_parser().print_help()
        return 0
    if arguments[0] in {"-h", "--help"}:
        build_parser().parse_args(arguments)
        return 0
    if arguments[0] == "image":
        return _run_image(arguments[1:])
    if arguments[0] == "report":
        return _run_report(arguments[1:])
    if command := _DELEGATED_COMMAND_BY_NAME.get(arguments[0]):
        return _run_delegated(command, arguments[1:])
    if arguments[0].startswith("-"):
        return _run_image(arguments)

    parser = build_parser()
    parser.error(f"unknown command: {arguments[0]}")


if __name__ == "__main__":
    sys.exit(main())
