from lx_anonymizer.ollama.ollama_llm_meta_extraction import (
    EnrichedMetadataExtractor,
    FrameDataProcessor,
    FrameSamplingOptimizer,
    MetadataCache,
    OllamaOptimizedExtractor,
    VideoMetadataEnricher,
)
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


def _extractor_stub(
    *,
    current_model=None,
    available_models=None,
    preferred_model=None,
    preferred_timeout=None,
):
    extractor = OllamaOptimizedExtractor.__new__(OllamaOptimizedExtractor)
    extractor.current_model = current_model
    extractor.available_models = available_models or []
    extractor.preferred_model = preferred_model
    extractor.preferred_timeout = preferred_timeout
    extractor.cache = MetadataCache()
    extractor.sensitive_meta = SensitiveMeta()
    return extractor


class _DummyOllama:
    def extract_metadata_smart_sampling(self, text):
        return None

    def _calculate_confidence(self, metadata):
        return 0.8 if metadata else 0.0


def test_metadata_cache_fifo_and_stats():
    cache = MetadataCache(max_size=2)
    m1 = SensitiveMeta(patient_first_name="A")
    m2 = SensitiveMeta(patient_first_name="B")
    m3 = SensitiveMeta(patient_first_name="C")

    cache.put("one", m1)
    cache.put("two", m2)
    assert cache.get("one") is m1
    assert cache.get("missing") is None

    cache.put("three", m3)

    assert cache.get("one") is None
    assert cache.get("two") is m2
    assert cache.get("three") is m3
    stats = cache.get_stats()
    assert stats["cache_size"] == 2
    assert stats["hit_count"] >= 3
    assert stats["miss_count"] >= 2
    assert 0 <= stats["hit_rate"] <= 1


def test_clean_json_response_strips_think_and_markdown():
    extractor = _extractor_stub()
    raw = """
    <think>hidden chain of thought</think>
    ```json
    {"patient_first_name":"Max","casenumber":"E123"}
    ```
    """
    assert (
        extractor._clean_json_response(raw)
        == '{"patient_first_name":"Max","casenumber":"E123"}'
    )


def test_clean_json_response_extracts_inline_json():
    extractor = _extractor_stub()
    raw = 'prefix text {"patient_last_name":"Muster"} trailing text'
    assert extractor._clean_json_response(raw) == '{"patient_last_name":"Muster"}'


def test_contains_patient_data_uses_threshold():
    extractor = _extractor_stub()
    assert extractor._contains_patient_data("short") is False
    assert extractor._contains_patient_data("Patient Name") is True
    assert extractor._contains_patient_data("foo bar baz qux lorem ipsum") is False


def test_calculate_confidence_scores_and_caps():
    extractor = _extractor_stub()
    high = extractor._calculate_confidence(
        {
            "patient_first_name": "Max",
            "patient_last_name": "Mustermann",
            "examination_date": "2024-01-01",
            "casenumber": "E 123",
            "patient_dob": "1990-01-01",
            "patient_gender_name": "male",
            "examiner_first_name": "A",
            "examiner_last_name": "B",
        }
    )
    low = extractor._calculate_confidence({"patient_first_name": "unknown"})
    assert high == 1.0
    assert low == 0.0


def test_get_fastest_available_model_honors_preferred_timeout():
    extractor = _extractor_stub(
        available_models=["llama3.2:1b", "qwen2.5:1.5b-instruct"],
        preferred_model="qwen2.5:1.5b-instruct",
        preferred_timeout=99,
    )
    model = extractor._get_fastest_available_model()
    assert model is not None
    assert model["name"] == "qwen2.5:1.5b-instruct"
    assert model["timeout"] == 99


def test_create_extraction_prompt_varies_by_model_size_and_truncates():
    small = _extractor_stub(current_model={"name": "llama3.2:1b"})
    big = _extractor_stub(current_model={"name": "deepseek-r1:1.5x"})  # not "small"
    text = "X" * 2000

    small_prompt = small._create_extraction_prompt(text)
    big_prompt = big._create_extraction_prompt(text)

    assert "Return exactly one JSON object" in small_prompt
    assert "Extract patient metadata" in small_prompt
    assert "Return ONLY JSON." in big_prompt
    assert len(small_prompt) < len(big_prompt)


def test_frame_sampling_optimizer_decisions_and_duplicate_skip():
    opt = FrameSamplingOptimizer()
    opt.register_processed_frame("dup", {"ok": True})

    assert opt.should_process_frame(0, 120, "x") is True
    assert opt.should_process_frame(118, 120, "x2") is True
    assert opt.should_process_frame(20, 120, "dup") is False
    assert opt.should_process_frame(10, 120, "new") is True  # %5 == 0
    assert opt.should_process_frame(11, 120, "new2") is False


def test_frame_sampling_strategy_buckets():
    opt = FrameSamplingOptimizer(max_frames=50)
    assert opt.get_sampling_strategy(40)["strategy"] == "dense"
    assert opt.get_sampling_strategy(150)["skip_factor"] == 2
    assert opt.get_sampling_strategy(500)["skip_factor"] == 5
    assert opt.get_sampling_strategy(5000)["max_samples"] == 50


def test_enriched_aggregate_ocr_texts_deduplicates_and_filters_short_lines():
    ext = EnrichedMetadataExtractor(_DummyOllama(), FrameSamplingOptimizer())
    frames = [
        {"ocr_text": "Name: Max\nID 1234\nabc"},
        {"ocr_text": "name max\nID-1234\nUntersuchung 2024"},
    ]
    ocr_texts = ["Name: Max\nNeue Zeile", None]

    result = ext._aggregate_ocr_texts(frames, ocr_texts)

    assert "Name: Max" in result
    assert "Neue Zeile" in result
    assert "Untersuchung 2024" in result
    assert result.count("Name: Max") == 1
    assert "abc" not in result


def test_enriched_temporal_analysis_detects_change_points():
    ext = EnrichedMetadataExtractor(_DummyOllama(), FrameSamplingOptimizer())
    frames = [
        {"ocr_text": "Patient Max Muster", "ocr_confidence": 0.9, "timestamp": 0.0},
        {"ocr_text": "Patient Max Muster", "ocr_confidence": 0.8, "timestamp": 1.0},
        {"ocr_text": "Totally different content", "ocr_confidence": 0.7, "timestamp": 2.0},
    ]

    temporal = ext._perform_temporal_analysis(frames, {})

    assert len(temporal["text_appearance_timeline"]) == 3
    assert temporal["change_points"] == [2]


def test_enriched_confidence_combines_sources():
    ext = EnrichedMetadataExtractor(_DummyOllama(), FrameSamplingOptimizer())
    scores = ext._calculate_enriched_confidence(
        {
            "llm_extracted": {"patient_first_name": "Max", "patient_last_name": "M"},
            "frame_context": {"quality_scores": [0.6, 0.8]},
            "temporal_analysis": {
                "text_appearance_timeline": [{"confidence": 0.4}, {"confidence": 0.8}]
            },
        }
    )
    assert scores["llm_confidence"] == 0.8
    assert scores["frame_quality_confidence"] == 0.7
    assert scores["temporal_stability_confidence"] == 0.6
    assert 0 < scores["overall_confidence"] <= 1


def test_frame_data_processor_normalizes_multiple_input_shapes():
    class FrameObj:
        def __init__(self):
            self.ocr_text = "from object"
            self.ocr_confidence = 0.75

    items = [
        {"ocr_text": "from dict", "has_text": True},
        FrameObj(),
        ("frame_bytes", "tuple text"),
    ]
    processed = FrameDataProcessor.process_coroutine_output(items)

    assert len(processed) == 3
    assert processed[0]["ocr_text"] == "from dict"
    assert processed[1]["ocr_confidence"] == 0.75
    assert processed[2]["ocr_text"] == "tuple text"
    assert processed[2]["has_text"] is True


def test_frame_data_processor_accepts_generators():
    generator = ({"ocr_text": f"t{i}"} for i in range(2))
    processed = FrameDataProcessor.process_coroutine_output(generator)
    assert [p["frame_index"] for p in processed] == [0, 1]


def test_video_metadata_merge_sources_uses_legacy_as_fallback_only():
    enricher = VideoMetadataEnricher.__new__(VideoMetadataEnricher)
    enriched = {
        "enriched_data": {"llm_extracted": {"patient_first_name": "LLM", "patient_dob": None}}
    }
    legacy = {"patient_first_name": "OCR", "patient_dob": "1990-01-01"}

    merged = enricher._merge_metadata_sources(enriched, legacy)

    assert "fallback_data" in merged
    assert "patient_first_name" not in merged["fallback_data"]
    assert merged["fallback_data"]["patient_dob"] == "1990-01-01"


def test_video_metadata_integration_stats_reports_sources_and_completeness():
    enricher = VideoMetadataEnricher.__new__(VideoMetadataEnricher)
    metadata = {
        "enriched_data": {"llm_extracted": {"patient_first_name": "Max"}},
        "legacy_data": {"patient_last_name": "Muster"},
        "fallback_data": {"patient_dob": "1990-01-01"},
    }

    stats = enricher._calculate_integration_stats(metadata)

    assert stats["data_sources_used"] == ["enriched_llm", "legacy_ocr", "fallback_data"]
    assert 0 < stats["data_completeness"] <= 1
