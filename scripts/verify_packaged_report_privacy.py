#!/usr/bin/env python3
"""Verify installed wheels in a real Celery solo worker (verification-only dependency).

Use an isolated interpreter outside a checkout. Inject provider and OCR/detector
boundaries; run real PDF reading, text redaction, PDF redaction and artifact API.
"""

from __future__ import annotations

import argparse
from contextlib import (
    AbstractContextManager,
    ExitStack,
    redirect_stderr,
    redirect_stdout,
)
from collections.abc import Callable, Iterator, Mapping
import hashlib
import importlib
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
from typing import Protocol, cast
from unittest.mock import patch
from uuid import uuid4


class PdfPage(Protocol):
    def insert_text(self, point: tuple[int, int], text: str) -> object: ...
    def get_text(self) -> str: ...


class PdfDocument(Protocol):
    def new_page(self, *, width: int, height: int) -> PdfPage: ...
    def save(self, filename: str) -> None: ...
    def close(self) -> None: ...
    def __iter__(self) -> Iterator[PdfPage]: ...


class PdfAPI(Protocol):
    def open(self, filename: str | None = None) -> PdfDocument: ...


class TaskResult(Protocol):
    state: str

    def get(self, *, timeout: int, propagate: bool) -> object: ...


class Task(Protocol):
    def delay(self, scenario: str) -> TaskResult: ...


class WorkerLogging(Protocol):
    def redirect_stdouts(self, *, loglevel: str) -> None: ...


class WorkerConfig(Protocol):
    def update(self, **kwargs: object) -> None: ...


class WorkerApp(Protocol):
    conf: WorkerConfig
    log: WorkerLogging

    def task(
        self, *, name: str
    ) -> Callable[[Callable[[str], dict[str, object]]], Task]: ...
    def close(self) -> None: ...


class AppFactory(Protocol):
    def __call__(self, name: str, *, broker: str, backend: str) -> WorkerApp: ...


class WorkerStart(Protocol):
    def __call__(
        self,
        app: WorkerApp,
        *,
        pool: str,
        concurrency: int,
        perform_ping_check: bool,
        loglevel: str,
        logfile: None,
    ) -> AbstractContextManager[object]: ...


def verify(site_packages: Path, captured: io.StringIO) -> dict[str, object]:
    # Narrow dynamic dependencies to the exact verification API used here.
    celery_factory = cast(
        AppFactory, getattr(importlib.import_module("celery"), "Celery")
    )
    worker_start = cast(
        WorkerStart,
        getattr(
            importlib.import_module("celery.contrib.testing.worker"), "start_worker"
        ),
    )
    from PIL import Image
    from lx_anonymizer.anonymization.anonymizer import Anonymizer
    from lx_anonymizer.config import settings
    from lx_anonymizer.report_reader import ReportReader
    from lx_anonymizer.report_contracts import ReportAnonymizationRequest
    from lx_anonymizer.sensitive_meta_interface import SensitiveMeta
    from lx_dtypes.models.contracts.report_anonymization import (
        REPORT_ANONYMIZATION_CONTRACT_VERSION,
    )

    origins = ("lx_anonymizer", "lx_dtypes", "lx_anonymizer._lx_anonymizer_native")
    for name in origins:
        origin = importlib.import_module(name).__file__
        assert origin and Path(origin).resolve().is_relative_to(site_packages)
    assert REPORT_ANONYMIZATION_CONTRACT_VERSION == "report_anonymization_v2"
    sentinel = "PrivatePatientPrivacySentinel"
    pdf_api = cast(PdfAPI, importlib.import_module("pymupdf"))
    app = celery_factory(
        "packaged_report_privacy", broker="memory://", backend="cache+memory://"
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        worker_hijack_root_logger=False,
        worker_redirect_stdouts=True,
    )
    handler = logging.StreamHandler(captured)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    outcomes: dict[str, str] = {}
    failures = {
        "provider_failure": "ProviderUnavailableError",
        "redaction_failure": "AnonymizationArtifactError",
    }
    with tempfile.TemporaryDirectory(prefix="packaged-report-privacy-") as directory:
        work = Path(directory)
        source = work / f"{sentinel}.pdf"
        doc = pdf_api.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text(
            (30, 50), f"Patient {sentinel} Synthetic. Untersuchung 01.01.2024."
        )
        doc.save(str(source))
        doc.close()
        original = source.read_bytes()

        class Provider:
            current_model: object = object()

            def __init__(self, scenario: str) -> None:
                self.scenario = scenario

            def extract_metadata(self, text: str) -> SensitiveMeta:
                assert sentinel in text
                if self.scenario == "provider_failure":
                    raise ConnectionError(f"provider response {sentinel} {source}")
                return SensitiveMeta(first_name=sentinel, last_name="Synthetic")

        def execute(scenario: str) -> dict[str, object]:
            output = work / scenario
            output.mkdir()
            request = ReportAnonymizationRequest(
                attempt_id=uuid4(),
                source_path=source,
                source_sha256=hashlib.sha256(original).hexdigest(),
                source_size_bytes=len(original),
                output_directory=output,
            )

            def detect(
                self: Anonymizer, image: Image.Image
            ) -> list[tuple[int, int, int, int]]:
                if scenario == "redaction_failure":
                    raise RuntimeError(f"detector payload {sentinel} {source}")
                return [(0, 0, image.width, image.height)]

            with ExitStack() as stack:
                stack.enter_context(patch.object(settings, "LLM_ENABLED", False))
                stack.enter_context(
                    patch.object(settings, "METADATA_MODEL_ORDER", ["llm"])
                )
                stack.enter_context(
                    patch.object(settings, "REPORT_LLM_MIN_TEXT_LENGTH", 1)
                )
                stack.enter_context(
                    patch(
                        "lx_anonymizer.anonymization.anonymizer.get_tessdata_path",
                        return_value=work,
                    )
                )
                stack.enter_context(
                    patch(
                        "lx_anonymizer.anonymization.anonymizer.convert_pdf_to_images",
                        return_value=[Image.new("RGB", (595, 842), "white")],
                    )
                )
                stack.enter_context(
                    patch.object(
                        Anonymizer, "_detect_sensitive_regions_with_fallback", detect
                    )
                )
                reader = ReportReader()
                reader.llm_available = True
                reader.llm_extractor = Provider(scenario)
                stack.enter_context(patch.object(settings, "LLM_ENABLED", True))
                result = reader.process_report(request)
            assert scenario == "success"
            assert result.contract_version == "report_anonymization_v2"
            assert result.extraction_evidence.outcome == "findings"
            assert result.extraction_evidence.backend == "configured_llm"
            assert result.artifact_path.is_file()
            assert (
                result.artifact_sha256
                == hashlib.sha256(result.artifact_path.read_bytes()).hexdigest()
            )
            assert sentinel not in result.anonymized_text
            redacted = pdf_api.open(str(result.artifact_path))
            try:
                assert sentinel not in "".join(page.get_text() for page in redacted)
            finally:
                redacted.close()
            assert len(list(output.iterdir())) == 1
            return {
                "scenario": scenario,
                "contract_version": result.contract_version,
                "artifact_valid": True,
            }

        task = app.task(name="privacy.process_report")(execute)
        with worker_start(
            app,
            pool="solo",
            concurrency=1,
            perform_ping_check=False,
            loglevel="DEBUG",
            logfile=None,
        ):
            with redirect_stdout(captured), redirect_stderr(captured):
                app.log.redirect_stdouts(loglevel="WARNING")
                print("privacy_worker_stdout_probe")
                print("privacy_worker_stderr_probe", file=sys.stderr)
                for scenario in ("success", "provider_failure", "redaction_failure"):
                    pending = task.delay(scenario)
                    value = pending.get(timeout=30, propagate=False)
                    if scenario == "success":
                        assert pending.state == "SUCCESS"
                        assert isinstance(value, Mapping)
                        assert (
                            cast(Mapping[str, object], value).get("artifact_valid")
                            is True
                        )
                    else:
                        assert pending.state == "FAILURE"
                        assert type(value).__name__ == failures[scenario]
                        assert not list((work / scenario).iterdir())
                    assert source.read_bytes() == original
                    outcomes[scenario] = pending.state
        app.close()
    root.removeHandler(handler)
    logs = captured.getvalue()
    assert sentinel not in logs, "private sentinel reached worker output"
    required = (
        "redaction_pdf_saved",
        "report_artifact_created",
        "report_metadata_provider_failed",
        "Traceback",
        "privacy_worker_stdout_probe",
        "privacy_worker_stderr_probe",
    )
    for event in required:
        assert event in logs, "required worker diagnostic missing"
    return {
        "status": "passed",
        "contract_version": REPORT_ANONYMIZATION_CONTRACT_VERSION,
        "installed_origins_verified": len(origins),
        "scenarios": outcomes,
        "safe_event_counts": {event: logs.count(event) for event in required},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", required=True, type=Path)
    args = parser.parse_args()
    captured = io.StringIO()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            result = verify(args.site_packages.resolve(), captured)
    except Exception as exc:
        # Never expose exception values or captured worker logs when checks fail.
        print(
            json.dumps(
                {"status": "failed", "error_class": type(exc).__name__}, sort_keys=True
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
