import argparse
import sys
from types import ModuleType, SimpleNamespace

import pytest

import lx_anonymizer
from lx_anonymizer import cli
from lx_anonymizer import settings as settings_module
from lx_anonymizer.config import Settings, settings


def test_build_parser_defaults_and_required_args() -> None:
    parser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    args = parser.parse_args(["-i", "input.png"])
    assert args.image == "input.png"
    assert args.east is None
    assert args.device == "olympus_cv_1500"
    assert args.validation is False
    assert args.min_confidence == 0.5
    assert args.width == 320
    assert args.height == 320
    assert args.verbose is False


def test_cli_main_success_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
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

    monkeypatch.setitem(sys.modules, "lx_anonymizer.main_with_reassembly", fake_pipeline_module)
    monkeypatch.setitem(sys.modules, "lx_anonymizer.setup.custom_logger", fake_logger_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lx-anonymizer",
            "-i",
            "input.pdf",
            "-east",
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
        ],
    )

    rc = cli.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert logger_calls == [True]
    assert pipeline_calls == [
        ("input.pdf", "model.pb", "default", True, 0.7, 640, 480)
    ]
    assert "done" in out


def test_cli_main_missing_dependency_exits_with_code_2(monkeypatch: pytest.MonkeyPatch) -> None:
    missing_pipeline_module = ModuleType("lx_anonymizer.main_with_reassembly")
    fake_logger_module = ModuleType("lx_anonymizer.setup.custom_logger")
    fake_logger_module.configure_global_logger = lambda **_: None  # type: ignore[attr-defined]

    # Importing `main` from this module should fail.
    monkeypatch.setitem(sys.modules, "lx_anonymizer.main_with_reassembly", missing_pipeline_module)
    monkeypatch.setitem(sys.modules, "lx_anonymizer.setup.custom_logger", fake_logger_module)
    monkeypatch.setattr(sys, "argv", ["lx-anonymizer", "-i", "input.pdf"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2


def test_package_getattr_resolves_frame_cleaner(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("lx_anonymizer.frame_cleaner")
    fake_cls = type("FakeFrameCleaner", (), {})
    fake_module.FrameCleaner = fake_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lx_anonymizer.frame_cleaner", fake_module)

    assert lx_anonymizer.__getattr__("FrameCleaner") is fake_cls


def test_package_getattr_resolves_report_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("lx_anonymizer.report_reader")
    fake_cls = type("FakeReportReader", (), {})
    fake_module.ReportReader = fake_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lx_anonymizer.report_reader", fake_module)

    assert lx_anonymizer.__getattr__("ReportReader") is fake_cls


def test_package_getattr_resolves_ollama_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = SimpleNamespace(name="ollama")

    def fake_import_module(name: str) -> object:
        assert name == "lx_anonymizer.ollama.ollama_llm"
        return sentinel

    monkeypatch.setattr(lx_anonymizer, "import_module", fake_import_module)
    assert lx_anonymizer.__getattr__("ollama_llm") is sentinel


def test_package_getattr_unknown_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        lx_anonymizer.__getattr__("does_not_exist")


def test_settings_module_reexports_symbols() -> None:
    assert settings_module.Settings is Settings
    assert settings_module.settings is settings
    assert sorted(settings_module.__all__) == ["Settings", "settings"]
