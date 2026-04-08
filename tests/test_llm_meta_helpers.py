from lx_anonymizer.llm.llm_extractor import (
    EnrichedMetadataExtractor,
    FrameDataProcessor,
    FrameSamplingOptimizer,
    LLMMetadataExtractor,
    MetadataCache,
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
    extractor = LLMMetadataExtractor.__new__(LLMMetadataExtractor)
    extractor.current_model = current_model
    extractor.available_models = available_models or []
    extractor.preferred_model = preferred_model
    extractor.preferred_timeout = preferred_timeout
    extractor.base_url = "http://127.0.0.1:8000"
    extractor.chat_endpoint = extractor._build_chat_endpoint()
    extractor.cache = MetadataCache()
    extractor.sensitive_meta = SensitiveMeta()
    return extractor


class _DummyLLM:
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
        available_models=["Qwen/Qwen3.5-9B", "Qwen/Qwen2.5-3B-Instruct"],
        preferred_model="Qwen/Qwen2.5-3B-Instruct",
        preferred_timeout=99,
    )
    model = extractor._get_fastest_available_model()
    assert model is not None
    assert model["name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert model["timeout"] == 99


def test_build_chat_endpoint_uses_openai_compatible_path_for_vllm():
    extractor = _extractor_stub()
    extractor.base_url = "http://127.0.0.1:8000/"
    extractor.provider = "vllm"
    assert (
        extractor._build_chat_endpoint() == "http://127.0.0.1:8000/v1/chat/completions"
    )


def test_extract_response_content_supports_vllm_choices_shape():
    extractor = _extractor_stub()
    content = extractor._extract_response_content(
        {"choices": [{"message": {"content": '{"patient_first_name":"Max"}'}}]}
    )
    assert content == '{"patient_first_name":"Max"}'


def test_get_fastest_available_model_uses_preferred_even_without_listing():
    extractor = _extractor_stub(
        available_models=[],
        preferred_model="Qwen/Qwen2.5-3B-Instruct",
        preferred_timeout=42,
    )
    model = extractor._get_fastest_available_model()
    assert model is not None
    assert model["name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert model["timeout"] == 42


def test_frame_sampling_optimizer_decisions_and_duplicate_skip():
    opt = FrameSamplingOptimizer()
    opt.register_processed_frame("dup", {"ok": True})

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
    ext = EnrichedMetadataExtractor(_DummyLLM(), FrameSamplingOptimizer())
    frames = [
        {"ocr_text": "Patient Max Muster", "ocr_confidence": 0.9, "timestamp": 0.0},
        {"ocr_text": "Patient Max Muster", "ocr_confidence": 0.8, "timestamp": 1.0},
        {
            "ocr_text": "Totally different content",
            "ocr_confidence": 0.7,
            "timestamp": 2.0,
        },
    ]

    temporal = ext._perform_temporal_analysis(frames, {})

    assert len(temporal["text_appearance_timeline"]) == 3
    assert temporal["change_points"] == [2]


def test_enriched_confidence_combines_sources():
    ext = EnrichedMetadataExtractor(_DummyLLM(), FrameSamplingOptimizer())
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


def test_video_metadata_enricher_keeps_existing_fallback_data():
    enricher = VideoMetadataEnricher()
    enricher.enriched_extractor.extract_from_frame_sequence = lambda *args, **kwargs: {
        "llm_extracted": {"patient_first_name": "LLM_Name"}
    }
    enricher.frame_processor.process_coroutine_output = lambda x: x

    result = enricher.enrich_from_multiple_sources(
        video_path="test.mp4",
        frame_samples=[{"dummy": "data"}],
        ocr_texts=[],
        existing_metadata={
            "patient_first_name": "OCR_Name_Typos",
            "patient_dob": "01.01.1990",
        },
    )

    assert result["enriched_data"]["llm_extracted"]["patient_first_name"] == "LLM_Name"
    assert result["fallback_data"]["patient_dob"] == "01.01.1990"
