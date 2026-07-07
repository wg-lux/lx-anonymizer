import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch
from lx_dtypes.models.contracts.llm_extractor import (
    LLMEnrichedMetadataPayload,
    LLMFrameContextPayload,
    LLMFrameDataPayload,
    LLMTextTimelineEntryPayload,
    LLMTemporalAnalysisPayload,
)
from lx_dtypes.models.contracts.llm_service import (
    LLMChatOllamaPayload,
    LLMChatResponsePayload,
)
from lx_dtypes.models.contracts.text_anonymization import LLMMetadataPayload
from lx_anonymizer.llm.llm_extractor import (
    AsyncMetadataWorker,
    EnrichedMetadataExtractor,
    FrameDataProcessor,
    FrameSamplingOptimizer,
    LLMChatRequestPayload,
    LLMFrameProcessorInput,
    LLMMetadataExtractor,
    MetadataCache,
    VideoMetadataEnricher,
    _LLMModelConfig,  # pyright: ignore[reportPrivateUsage]
    _parse_ollama_model_names,  # pyright: ignore[reportPrivateUsage]
)
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


def _extractor_stub(
    *,
    current_model: _LLMModelConfig | Mapping[str, object] | None = None,
    available_models: list[str] | None = None,
    preferred_model: str | None = None,
    preferred_timeout: int | None = None,
    provider: str = "ollama",
) -> LLMMetadataExtractor:
    extractor = LLMMetadataExtractor.__new__(LLMMetadataExtractor)
    extractor.provider = provider
    extractor.current_model = (
        _LLMModelConfig.model_validate(current_model)
        if current_model is not None and not isinstance(current_model, _LLMModelConfig)
        else current_model
    )
    extractor.available_models = available_models or []
    extractor.available_models_retry = False
    extractor.preferred_model = preferred_model
    extractor.preferred_timeout = preferred_timeout
    extractor.base_url = (
        "http://127.0.0.1:11434" if provider == "ollama" else "http://127.0.0.1:8000"
    )
    extractor.chat_endpoint = extractor._build_chat_endpoint()  # pyright: ignore[reportPrivateUsage]
    extractor.cache = MetadataCache()
    extractor.sensitive_meta = SensitiveMeta()
    return extractor


class _DummyLLM(LLMMetadataExtractor):
    def __init__(self) -> None:
        pass

    def extract_metadata_smart_sampling(
        self, text: str, confidence_threshold: float = 0.7
    ) -> SensitiveMeta | None:
        return None

    def calculate_confidence(self, metadata: LLMMetadataPayload) -> float:
        return self._calculate_confidence(metadata)

    def _calculate_confidence(self, metadata: LLMMetadataPayload) -> float:
        return 0.8 if metadata else 0.0


class _ThreadRecordingExtractor(LLMMetadataExtractor):
    delay: float
    called_thread: int | None

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.called_thread = None

    def extract_metadata(self, text: str) -> SensitiveMeta | None:
        self.called_thread = threading.get_ident()
        if self.delay:
            time.sleep(self.delay)
        return SensitiveMeta(first_name=text)


class _StubEnrichedExtractor(EnrichedMetadataExtractor):
    def __init__(self) -> None:
        pass

    def extract_from_frame_sequence(
        self, frames_data: Sequence[LLMFrameDataPayload], ocr_texts: list[str]
    ) -> LLMEnrichedMetadataPayload:
        return LLMEnrichedMetadataPayload(
            llm_extracted=LLMMetadataPayload(first_name="LLM_Name")
        )


class _StubFrameProcessor(FrameDataProcessor):
    @staticmethod
    def process_coroutine_output(
        coroutine_result: Iterable[LLMFrameProcessorInput],
    ) -> list[LLMFrameDataPayload]:
        return [LLMFrameDataPayload.model_validate(item) for item in coroutine_result]


class _ResponseStub:
    status_code: int
    text: str
    _payload: Mapping[str, object]

    def __init__(self, payload: Mapping[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self) -> Mapping[str, object]:
        return self._payload


def test_model_priority_prefers_custom_gemma_profile():
    extractor = _extractor_stub(available_models=["gemma4:e2b"])
    extractor._initialize_best_model()  # pyright: ignore[reportPrivateUsage]

    assert extractor.current_model is not None
    assert extractor.current_model.name == "gemma4:e2b"


def test_model_initialization_does_not_pick_non_allowlisted_fallback():
    extractor = _extractor_stub(available_models=["deepseek-r1:1.5b"])
    extractor._initialize_best_model()  # pyright: ignore[reportPrivateUsage]

    assert extractor.current_model is None


def test_parse_ollama_model_names_accepts_metadata_rich_tags_payload():
    payload: dict[str, object] = {
        "models": [
            {
                "name": "deepseek-r1:1.5b",
                "model": "deepseek-r1:1.5b",
                "modified_at": "2026-03-31T11:40:22.609640149+02:00",
                "size": 1117322768,
                "digest": ("e0979632db5a88d1a53884cb8cf34a1365aabaa6801c4e36f3a6c2d7"),
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "qwen2",
                    "families": ["qwen2"],
                    "parameter_size": "1.8B",
                    "quantization_level": "Q4_K_M",
                    "embedding_length": 1536,
                },
                "capabilities": ["completion", "thinking"],
            }
        ]
    }

    assert _parse_ollama_model_names(payload) == ["deepseek-r1:1.5b"]


def test_parse_ollama_model_names_accepts_model_field_without_name():
    payload: dict[str, object] = {"models": [{"model": "gemma4:e2b"}]}

    assert _parse_ollama_model_names(payload) == ["gemma4:e2b"]


def test_parse_ollama_model_names_rejects_entries_without_identifier():
    payload: dict[str, object] = {"models": [{"size": 1117322768}]}

    with pytest.raises(ValidationError):
        _parse_ollama_model_names(payload)


def test_check_available_models_uses_metadata_rich_ollama_tags_parser(
    monkeypatch: MonkeyPatch,
) -> None:
    payload: dict[str, object] = {
        "models": [
            {
                "name": "deepseek-r1:1.5b",
                "model": "deepseek-r1:1.5b",
                "modified_at": "2026-03-31T11:40:22.609640149+02:00",
                "size": 1117322768,
                "digest": "e0979632db5a",
                "details": {"embedding_length": 1536},
                "capabilities": ["completion", "thinking"],
            }
        ]
    }

    def fake_get(url: str, timeout: float) -> _ResponseStub:
        assert url == "http://127.0.0.1:11434/api/tags"
        assert timeout == 5
        return _ResponseStub(payload)

    monkeypatch.setattr("lx_anonymizer.llm.llm_extractor.requests.get", fake_get)
    extractor = _extractor_stub()

    assert extractor._check_available_models() == ["deepseek-r1:1.5b"]  # pyright: ignore[reportPrivateUsage]


def test_extraction_prompt_uses_sensitive_meta_subset_only():
    extractor = _extractor_stub(current_model={"name": "gemma4:e2b", "timeout": 120})

    prompt = extractor._create_extraction_prompt("Patient Max")  # pyright: ignore[reportPrivateUsage]
    fast_prompt = extractor._create_fast_extraction_prompt("Patient Max")  # pyright: ignore[reportPrivateUsage]

    for key in (
        "first_name",
        "last_name",
        "dob",
        "casenumber",
        "examination_date",
    ):
        assert key in prompt
        assert key in fast_prompt

    for excluded in (
        "gender",
        "examiner_first_name",
        "examiner_last_name",
        "examination_time",
    ):
        assert excluded not in prompt
        assert excluded not in fast_prompt


def test_async_metadata_worker_submit_runs_on_worker_thread():
    caller_thread = threading.get_ident()
    extractor = _ThreadRecordingExtractor()

    with AsyncMetadataWorker(extractor=extractor) as worker:
        future = worker.submit("Max")
        metadata = future.result(timeout=2)

    assert metadata is not None
    assert metadata.first_name == "Max"
    assert extractor.called_thread is not None
    assert extractor.called_thread != caller_thread


def test_async_metadata_worker_timeout_returns_none():
    extractor = _ThreadRecordingExtractor(delay=0.2)

    with AsyncMetadataWorker(extractor=extractor) as worker:
        metadata = worker.extract_metadata("Max", timeout=0.01)

    assert metadata is None


def test_metadata_cache_fifo_and_stats():
    cache = MetadataCache(max_size=2)
    m1 = SensitiveMeta(first_name="A")
    m2 = SensitiveMeta(first_name="B")
    m3 = SensitiveMeta(first_name="C")

    cache.put("one", m1)
    cache.put("two", m2)
    assert cache.get("one") is m1
    assert cache.get("missing") is None

    cache.put("three", m3)

    assert cache.get("one") is None
    assert cache.get("two") is m2
    assert cache.get("three") is m3
    stats = cache.get_stats()
    assert stats.cache_size == 2
    assert stats.hit_count >= 3
    assert stats.miss_count >= 2
    assert 0 <= stats.hit_rate <= 1


def test_clean_json_response_strips_think_and_markdown():
    extractor = _extractor_stub()
    raw = """
    <think>hidden chain of thought</think>
    ```json
    {"first_name":"Max","casenumber":"E123"}
    ```
    """
    assert (
        extractor._clean_json_response(raw)  # pyright: ignore[reportPrivateUsage]
        == '{"first_name":"Max","casenumber":"E123"}'
    )


def test_clean_json_response_strips_case_insensitive_unclosed_think():
    extractor = _extractor_stub()
    raw = '<THINK>hidden chain of thought {"first_name":"Max"}'

    assert extractor._clean_json_response(raw) == '{"first_name":"Max"}'  # pyright: ignore[reportPrivateUsage]


def test_clean_json_response_extracts_inline_json():
    extractor = _extractor_stub()
    raw = 'prefix text {"last_name":"Muster"} trailing text'
    assert extractor._clean_json_response(raw) == '{"last_name":"Muster"}'  # pyright: ignore[reportPrivateUsage]


def test_extract_metadata_ollama_payload_uses_json_format_and_8k_context(
    monkeypatch: MonkeyPatch,
) -> None:
    extractor = _extractor_stub(
        current_model={"name": "gemma4:e2b", "timeout": 120},
        available_models=["gemma4:e2b"],
    )
    captured: dict[str, LLMChatRequestPayload] = {}

    def fake_request(payload: LLMChatRequestPayload) -> LLMChatResponsePayload:
        captured["payload"] = payload
        return LLMChatResponsePayload.model_validate(
            {"message": {"content": '{"first_name":"Max"}'}}
        )

    monkeypatch.setattr(extractor, "_make_api_request", fake_request)

    metadata = extractor.extract_metadata("Patient Max")

    assert metadata is not None
    payload = captured["payload"]
    assert isinstance(payload, LLMChatOllamaPayload)
    assert payload.format == "json"
    assert payload.options.temperature == 0
    assert payload.options.num_ctx == 8192


def test_smart_sampling_ollama_payload_uses_json_format_and_8k_context(
    monkeypatch: MonkeyPatch,
) -> None:
    extractor = _extractor_stub(
        current_model={"name": "gemma4:e2b", "timeout": 120},
        available_models=["gemma4:e2b"],
    )
    captured: dict[str, LLMChatRequestPayload] = {}

    def fake_request(payload: LLMChatRequestPayload) -> LLMChatResponsePayload:
        captured["payload"] = payload
        return LLMChatResponsePayload.model_validate(
            {
                "message": {
                    "content": (
                        '{"first_name":"Max","last_name":"Muster",'
                        '"dob":"1980-01-01","casenumber":"E123",'
                        '"examination_date":"2024-01-01"}'
                    )
                }
            }
        )

    monkeypatch.setattr(extractor, "_make_api_request", fake_request)

    metadata = extractor.extract_metadata_smart_sampling(
        "Patient Max Muster geboren 01.01.1980 Untersuchung 01.01.2024 Fall E123"
    )

    assert metadata is not None
    payload = captured["payload"]
    assert isinstance(payload, LLMChatOllamaPayload)
    assert payload.format == "json"
    assert payload.options.temperature == 0
    assert payload.options.num_ctx == 8192


def test_contains_patient_data_uses_threshold():
    extractor = _extractor_stub()
    assert extractor._contains_patient_data("short") is False  # pyright: ignore[reportPrivateUsage]
    assert extractor._contains_patient_data("Patient Name") is True  # pyright: ignore[reportPrivateUsage]
    assert extractor._contains_patient_data("foo bar baz qux lorem ipsum") is False  # pyright: ignore[reportPrivateUsage]


def test_calculate_confidence_scores_and_caps():
    extractor = _extractor_stub()
    high = extractor._calculate_confidence(  # pyright: ignore[reportPrivateUsage]
        LLMMetadataPayload(
            first_name="Max",
            last_name="Mustermann",
            examination_date="2024-01-01",
            casenumber="E 123",
            dob="1990-01-01",
            gender="male",
            examiner_first_name="A",
            examiner_last_name="B",
        )
    )
    low = extractor._calculate_confidence(LLMMetadataPayload(first_name="unknown"))  # pyright: ignore[reportPrivateUsage]
    assert high == 1.0
    assert low == 0.0


def test_get_fastest_available_model_honors_preferred_timeout():
    extractor = _extractor_stub(
        available_models=["Qwen/Qwen3.5-9B", "Qwen/Qwen2.5-3B-Instruct"],
        preferred_model="Qwen/Qwen2.5-3B-Instruct",
        preferred_timeout=99,
    )
    model = extractor._get_fastest_available_model()  # pyright: ignore[reportPrivateUsage]
    assert model is not None
    assert model.name == "Qwen/Qwen2.5-3B-Instruct"
    assert model.timeout == 99


def test_build_chat_endpoint_uses_openai_compatible_path_for_vllm():
    extractor = _extractor_stub()
    extractor.base_url = "http://127.0.0.1:8000/"
    extractor.provider = "vllm"
    assert (
        extractor._build_chat_endpoint() == "http://127.0.0.1:8000/v1/chat/completions"  # pyright: ignore[reportPrivateUsage]
    )


def test_extract_response_content_supports_vllm_choices_shape():
    extractor = _extractor_stub()
    content = extractor._extract_response_content(  # pyright: ignore[reportPrivateUsage]
        LLMChatResponsePayload.model_validate(
            {"choices": [{"message": {"content": '{"first_name":"Max"}'}}]}
        )
    )
    assert content == '{"first_name":"Max"}'


def test_get_fastest_available_model_uses_preferred_even_without_listing():
    extractor = _extractor_stub(
        available_models=[],
        preferred_model="Qwen/Qwen2.5-3B-Instruct",
        preferred_timeout=42,
    )
    model = extractor._get_fastest_available_model()  # pyright: ignore[reportPrivateUsage]
    assert model is not None
    assert model.name == "Qwen/Qwen2.5-3B-Instruct"
    assert model.timeout == 42


def test_frame_sampling_optimizer_decisions_and_duplicate_skip():
    opt = FrameSamplingOptimizer()
    opt.register_processed_frame("dup", LLMFrameDataPayload(has_text=True))

    assert opt.should_process_frame(0, 120, "x") is True
    assert opt.should_process_frame(118, 120, "x2") is True
    assert opt.should_process_frame(20, 120, "dup") is False
    assert opt.should_process_frame(10, 120, "new") is True
    assert opt.should_process_frame(11, 120, "new2") is False


def test_frame_sampling_strategy_buckets():
    opt = FrameSamplingOptimizer(max_frames=50)
    assert opt.get_sampling_strategy(40)["strategy"] == "dense"
    assert opt.get_sampling_strategy(150)["skip_factor"] == 2
    assert opt.get_sampling_strategy(500)["skip_factor"] == 5
    assert opt.get_sampling_strategy(5000)["max_samples"] == 50


def test_enriched_aggregate_ocr_texts_deduplicates_and_filters_short_lines():
    ext = EnrichedMetadataExtractor(_DummyLLM(), FrameSamplingOptimizer())
    frames = [
        LLMFrameDataPayload(ocr_text="Name: Max\nID 1234\nabc"),
        LLMFrameDataPayload(ocr_text="name max\nID-1234\nUntersuchung 2024"),
    ]
    ocr_texts = ["Name: Max\nNeue Zeile"]

    result = ext._aggregate_ocr_texts(frames, ocr_texts)  # pyright: ignore[reportPrivateUsage]

    assert "Name: Max" in result
    assert "Neue Zeile" in result
    assert "Untersuchung 2024" in result
    assert result.count("Name: Max") == 1
    assert "abc" not in result


def test_enriched_temporal_analysis_detects_change_points():
    ext = EnrichedMetadataExtractor(_DummyLLM(), FrameSamplingOptimizer())
    frames = [
        LLMFrameDataPayload(
            ocr_text="Patient Max Muster", ocr_confidence=0.9, timestamp=0.0
        ),
        LLMFrameDataPayload(
            ocr_text="Patient Max Muster", ocr_confidence=0.8, timestamp=1.0
        ),
        LLMFrameDataPayload(
            ocr_text="Totally different content", ocr_confidence=0.7, timestamp=2.0
        ),
    ]

    temporal = ext._perform_temporal_analysis(frames, LLMEnrichedMetadataPayload())  # pyright: ignore[reportPrivateUsage]

    assert len(temporal.text_appearance_timeline) == 3
    assert temporal.change_points == [2]


def test_enriched_confidence_combines_sources():
    ext = EnrichedMetadataExtractor(_DummyLLM(), FrameSamplingOptimizer())
    scores = ext._calculate_enriched_confidence(  # pyright: ignore[reportPrivateUsage]
        LLMEnrichedMetadataPayload(
            llm_extracted=LLMMetadataPayload(first_name="Max", last_name="M"),
            frame_context=LLMFrameContextPayload(quality_scores=[0.6, 0.8]),
            temporal_analysis=LLMTemporalAnalysisPayload(
                text_appearance_timeline=[
                    LLMTextTimelineEntryPayload(
                        frame_index=0,
                        timestamp=0.0,
                        text_snippet="a",
                        confidence=0.4,
                    ),
                    LLMTextTimelineEntryPayload(
                        frame_index=1,
                        timestamp=1.0,
                        text_snippet="b",
                        confidence=0.8,
                    ),
                ]
            ),
        )
    )
    assert scores["llm_confidence"] == 0.8
    assert scores["frame_quality_confidence"] == 0.7
    assert scores["temporal_stability_confidence"] == 0.6
    assert 0 < scores["overall_confidence"] <= 1


def test_frame_data_processor_normalizes_multiple_input_shapes():
    items: list[LLMFrameProcessorInput] = [
        {"ocr_text": "from dict", "has_text": True},
        {"ocr_text": "from mapping", "ocr_confidence": 0.75},
        ("frame_bytes", "tuple text"),
    ]
    processed = FrameDataProcessor.process_coroutine_output(items)

    assert len(processed) == 3
    assert processed[0].ocr_text == "from dict"
    assert processed[1].ocr_text == "from mapping"
    assert processed[1].ocr_confidence == 0.75
    assert processed[2].ocr_text == "tuple text"
    assert processed[2].has_text is True


def test_frame_data_processor_accepts_generators():
    generator: Iterable[LLMFrameProcessorInput] = (
        {"ocr_text": f"t{i}"} for i in range(2)
    )
    processed = FrameDataProcessor.process_coroutine_output(generator)
    assert [p.frame_index for p in processed] == [0, 1]


def test_video_metadata_enricher_keeps_existing_fallback_data():
    enricher = VideoMetadataEnricher()
    enricher.enriched_extractor = _StubEnrichedExtractor()
    enricher.frame_processor = _StubFrameProcessor()

    result = enricher.enrich_from_multiple_sources(
        video_path="test.mp4",
        frame_samples=[{"dummy": "data"}],
        ocr_texts=[],
        existing_metadata={
            "first_name": "OCR_Name_Typos",
            "dob": "01.01.1990",
        },
    )

    enriched_data = cast(dict[str, object], result["enriched_data"])
    llm_extracted = cast(dict[str, object], enriched_data["llm_extracted"])
    fallback_data = cast(dict[str, object], result["fallback_data"])
    assert llm_extracted["first_name"] == "LLM_Name"
    assert fallback_data["dob"] == "01.01.1990"
