"""Provider and NER diagnostics must not disclose clinical input or failures."""

import hashlib
import logging
from importlib import import_module
from typing import cast

import pytest
import requests
import spacy
from spacy.pipeline import EntityRuler
from lx_dtypes.models.contracts.llm_service import LLMChatResponsePayload

from lx_anonymizer.config import settings
from lx_anonymizer.llm.llm_extractor import LLMMetadataExtractor, MetadataCache
from lx_anonymizer.ner.spacy_extractor import clean_date_value
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


SENTINEL = "PrivatePatientSentinel"


@pytest.mark.parametrize("failure", [False, True])
def test_provider_sampling_logs_omit_content(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: bool
) -> None:
    monkeypatch.setattr(settings, "LLM_ENABLED", True)

    def available_models(self: LLMMetadataExtractor) -> list[str]:
        return ["synthetic-model"]

    monkeypatch.setattr(
        LLMMetadataExtractor, "_check_available_models", available_models
    )
    extractor = LLMMetadataExtractor(preferred_model="synthetic-model")
    model = extractor.current_model
    assert model is not None
    monkeypatch.setattr(extractor, "_get_fastest_available_model", lambda: model)
    extractor.current_model = None  # Exercise sampling rather than same-model shortcut.

    def request(*args: object, **kwargs: object) -> LLMChatResponsePayload:
        if failure:
            raise requests.ConnectionError(f"{SENTINEL} /private/report.pdf")
        return LLMChatResponsePayload.model_validate(
            {
                "message": {
                    "content": '{"first_name":"PrivatePatientSentinel",'
                    '"last_name":"Synthetic","dob":"1980-01-01",'
                    '"casenumber":"123","examination_date":"2024-01-01"}'
                }
            }
        )

    monkeypatch.setattr(extractor, "_make_api_request", request)
    with caplog.at_level(logging.DEBUG, logger="lx_anonymizer.llm.llm_extractor"):
        result = extractor.extract_metadata_smart_sampling(
            f"Patient {SENTINEL} geboren 01.01.1980 Untersuchung 01.01.2024"
        )
    assert (result is None) == failure
    assert SENTINEL not in caplog.text
    assert "/private/report.pdf" not in caplog.text
    assert "1980-01-01" not in caplog.text
    assert "sampling" in caplog.text.lower()


def test_cache_and_date_logs_omit_content(caplog: pytest.LogCaptureFixture) -> None:
    cache = MetadataCache()
    with caplog.at_level(logging.DEBUG, logger="lx_anonymizer.llm.llm_extractor"):
        cache.put(SENTINEL, SensitiveMeta())
        assert cache.get(SENTINEL) is not None
    with caplog.at_level(logging.DEBUG, logger="lx_anonymizer.ner.spacy_extractor"):
        assert clean_date_value(SENTINEL) is None
    assert SENTINEL not in caplog.text
    assert hashlib.sha256(SENTINEL.encode()).hexdigest() not in caplog.text
    assert "Metadata cache hit" in caplog.text
    assert "Date parsing failed" in caplog.text


def test_spacy_ner_omits_entities_and_stdout(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = import_module("lx_anonymizer.ner.spacy_NER")
    nlp = spacy.blank("de")
    ruler = cast(EntityRuler, nlp.add_pipe("entity_ruler"))
    ruler.add_patterns([{"label": "PER", "pattern": SENTINEL}])
    monkeypatch.setattr(module, "nlp", nlp)
    result = module.spacy_NER_German(SENTINEL)
    assert result and result[0][0] == SENTINEL
    assert SENTINEL not in caplog.text
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out + captured.err
    assert "entity_count=1" in caplog.text


def test_provider_connection_failure_omits_endpoint_and_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "LLM_ENABLED", True)

    def available_models(self: LLMMetadataExtractor) -> list[str]:
        return ["synthetic-model"]

    def post(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError(f"{SENTINEL} /private/report.pdf")

    monkeypatch.setattr(
        LLMMetadataExtractor, "_check_available_models", available_models
    )
    monkeypatch.setattr(requests, "post", post)
    extractor = LLMMetadataExtractor(
        preferred_model="synthetic-model", base_url="http://127.0.0.1:11434"
    )
    with caplog.at_level(logging.DEBUG, logger="lx_anonymizer.llm.llm_extractor"):
        assert (
            extractor.extract_metadata(
                f"Patient {SENTINEL} geboren 01.01.1980 Untersuchung 01.01.2024"
            )
            is None
        )
    assert "LLM connection failed (ConnectionError)" in caplog.text
    for private in (SENTINEL, "127.0.0.1:11434", "/private/report.pdf"):
        assert private not in caplog.text
