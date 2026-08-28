import argparse
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

import lx_anonymizer
from lx_anonymizer import cli
from lx_anonymizer import settings as settings_module
from lx_anonymizer.config import Settings, settings


def test_build_parser_lists_commands() -> None:
    parser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    help_text = parser.format_help()
    assert "image" in help_text
    assert "report" in help_text
    assert "evaluate-midi-b" in help_text
    assert "train-phi" in help_text


def test_build_image_parser_defaults_and_legacy_input() -> None:
    args = cli.build_image_parser().parse_args(["-i", "input.png"])
    assert args.input is None
    assert args.legacy_input == Path("input.png")
    assert args.east is None
    assert args.device == "olympus_cv_1500"
    assert args.validation is False
    assert args.min_confidence == 0.5
    assert args.width == 320
    assert args.height == 320
    assert args.verbose is False


def test_cli_main_success_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline_calls: list[tuple[object, ...]] = []
    logger_calls: list[bool] = []

    fake_pipeline_module = ModuleType("lx_anonymizer.main_with_reassembly")

    def fake_pipeline_main(*args: object) -> str:
        pipeline_calls.append(args)
        return "done"

    fake_pipeline_module.main = fake_pipeline_main  # type: ignore[attr-defined]

    fake_logger_module = ModuleType("lx_anonymizer.setup.custom_logger")

    def fake_configure_global_logger(*, verbose: bool) -> None:
        logger_calls.append(verbose)

    fake_logger_module.configure_global_logger = fake_configure_global_logger  # type: ignore[attr-defined]

    monkeypatch.setitem(
        sys.modules, "lx_anonymizer.main_with_reassembly", fake_pipeline_module
    )
    monkeypatch.setitem(
        sys.modules, "lx_anonymizer.setup.custom_logger", fake_logger_module
    )
    rc = cli.main(
        [
            "image",
            "input.pdf",
            "--east",
            "model.pb",
            "-d",
            "default",
            "-V",
            "-c",
            "0.7",
            "-w",
            "640",
            "-e",
            "480",
            "-v",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert logger_calls == [True]
    assert pipeline_calls == [
        ("input.pdf", Path("model.pb"), "default", True, 0.7, 640, 480)
    ]
    assert "done" in out


def test_cli_main_missing_dependency_exits_with_code_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_pipeline_module = ModuleType("lx_anonymizer.main_with_reassembly")
    fake_logger_module = ModuleType("lx_anonymizer.setup.custom_logger")
    fake_logger_module.configure_global_logger = lambda **_: None  # type: ignore[attr-defined]

    # Importing `main` from this module should fail.
    monkeypatch.setitem(
        sys.modules, "lx_anonymizer.main_with_reassembly", missing_pipeline_module
    )
    monkeypatch.setitem(
        sys.modules, "lx_anonymizer.setup.custom_logger", fake_logger_module
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["-i", "input.pdf"])
    assert exc_info.value.code == 2


def test_cli_delegates_utility_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake_module = ModuleType("lx_anonymizer.evaluation.midi_b")

    def fake_main(argv: object = None) -> int:
        calls.append(argv)
        return 7

    fake_module.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lx_anonymizer.evaluation.midi_b", fake_module)

    assert cli.main(["evaluate-midi-b", "--help"]) == 7
    assert calls == [["--help"]]


def test_report_command_builds_typed_attempt_and_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4\nexample")
    output_directory = tmp_path / "attempt"
    captured: dict[str, object] = {}

    fake_contracts = ModuleType("lx_anonymizer.report_contracts")

    class FakeOptions:
        def __init__(self, **values: object) -> None:
            captured["options"] = values

    class FakeRequest:
        def __init__(self, **values: object) -> None:
            captured["request"] = values

    fake_contracts.ReportAnonymizationOptions = FakeOptions  # type: ignore[attr-defined]
    fake_contracts.ReportAnonymizationRequest = FakeRequest  # type: ignore[attr-defined]

    fake_reader = ModuleType("lx_anonymizer.report_reader")

    class FakeResult:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"artifact_path": str(output_directory / "candidate.pdf")}

    class FakeReportReader:
        def __init__(self, *, locale: str | None) -> None:
            captured["locale"] = locale

        def process_report(self, request: object) -> FakeResult:
            captured["processed_request"] = request
            return FakeResult()

    fake_reader.ReportReader = FakeReportReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lx_anonymizer.report_contracts", fake_contracts)
    monkeypatch.setitem(sys.modules, "lx_anonymizer.report_reader", fake_reader)

    assert (
        cli.main(
            [
                "report",
                str(source),
                "--output-directory",
                str(output_directory),
                "--locale",
                "de_DE",
                "--ensemble",
                "--no-llm",
            ]
        )
        == 0
    )

    request_values = cast(dict[str, object], captured["request"])
    assert request_values["source_path"] == source.resolve()
    assert request_values["source_size_bytes"] == source.stat().st_size
    assert request_values["output_directory"] == output_directory.resolve()
    assert captured["options"] == {
        "use_ensemble": True,
        "verbose": True,
        "use_llm": False,
    }
    assert captured["locale"] == "de_DE"
    assert json.loads(capsys.readouterr().out)["artifact_path"].endswith(
        "candidate.pdf"
    )


@pytest.mark.parametrize("suffix", [".png", ".txt", ""])
def test_report_command_rejects_non_pdf_input(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"report{suffix}"
    source.write_bytes(b"not a PDF")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["report", str(source), "-o", str(tmp_path / "output")])

    assert exc_info.value.code == 2


def test_package_getattr_resolves_frame_cleaner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("lx_anonymizer.frame_cleaner")
    fake_cls = type("FakeFrameCleaner", (), {})
    fake_module.FrameCleaner = fake_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lx_anonymizer.frame_cleaner", fake_module)

    assert lx_anonymizer.__getattr__("FrameCleaner") is fake_cls


def test_package_getattr_resolves_report_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("lx_anonymizer.report_reader")
    fake_cls = type("FakeReportReader", (), {})
    fake_module.ReportReader = fake_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lx_anonymizer.report_reader", fake_module)

    assert lx_anonymizer.__getattr__("ReportReader") is fake_cls


def test_package_getattr_unknown_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        lx_anonymizer.__getattr__("does_not_exist")


def test_settings_module_reexports_symbols() -> None:
    assert settings_module.Settings is Settings
    assert settings_module.settings is settings
    assert sorted(settings_module.__all__) == ["Settings", "settings"]
