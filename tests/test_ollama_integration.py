"""
Comprehensive tests for Optimized Ollama LLM Metadata Extraction.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch
from lx_anonymizer.ollama.ollama_llm_meta_extraction import (
    EnrichedMetadataExtractor,
    FrameSamplingOptimizer,
    OllamaOptimizedExtractor,
    VideoMetadataEnricher,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


class TestOllamaOptimizedExtractor(unittest.TestCase):
    """Test cases for the core OllamaOptimizedExtractor class"""

    def setUp(self):
        self.base_url = "http://localhost:11434"
        self.requests_patcher = patch(
            "lx_anonymizer.ollama.ollama_llm_meta_extraction.requests"
        )
        self.mock_requests = self.requests_patcher.start()

        self.available_models = [
            "llama3.2:1b",
            "qwen2.5:1.5b-instruct",
            "phi3.5:3.8b-mini-instruct-q4_K_M",
        ]
        self.models_patcher = patch.object(
            OllamaOptimizedExtractor,
            "_check_available_models",
            return_value=self.available_models,
        )
        self.mock_check_models = self.models_patcher.start()

        self.valid_json_response = json.dumps(
            {
                "patient_first_name": "Max",
                "patient_last_name": "Mustermann",
                "examination_date": "15.01.2024",
                "casenumber": "CASE-123",
            }
        )

        self.mock_response = MagicMock()
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {
            "message": {"content": self.valid_json_response},
            "eval_count": 50,
        }
        self.mock_requests.post.return_value = self.mock_response

    def tearDown(self):
        self.requests_patcher.stop()
        self.models_patcher.stop()

    def test_extract_metadata_success(self):
        """Test successful metadata extraction from text."""
        extractor = OllamaOptimizedExtractor()
        text = "Patient Max Mustermann, examined on 15.01.2024"

        result = extractor.extract_metadata(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["patient_first_name"], "Max")
        self.assertEqual(result["patient_last_name"], "Mustermann")

        call_args = self.mock_requests.post.call_args
        self.assertEqual(call_args[0][0], f"{self.base_url}/api/chat")

        payload = call_args[1]["json"]
        self.assertEqual(payload["model"], extractor.current_model["name"])
        self.assertIn(payload["model"], self.available_models)

    def test_initialization_selects_highest_priority_model(self):
        """Test that the extractor picks a valid model from the available list."""
        extractor = OllamaOptimizedExtractor(base_url=self.base_url)
        self.assertIsNotNone(extractor.current_model)
        self.assertIn(extractor.current_model["name"], self.available_models)

    def test_fallback_logic_on_bad_json(self):
        """Test fallback to next model on bad JSON."""
        extractor = OllamaOptimizedExtractor()

        bad_response = MagicMock()
        bad_response.status_code = 200
        bad_response.json.return_value = {"message": {"content": "I am not JSON"}}

        good_response = MagicMock()
        good_response.status_code = 200
        good_response.json.return_value = {
            "message": {"content": self.valid_json_response}
        }

        self.mock_requests.post.side_effect = [bad_response, good_response]

        result = extractor.extract_metadata("Some text with Patient info")

        self.assertEqual(result["patient_first_name"], "Max")
        self.assertEqual(self.mock_requests.post.call_count, 2)

    def test_smart_sampling_short_circuit(self):
        """Test that smart sampling returns early if confidence is high."""
        extractor = OllamaOptimizedExtractor()

        high_conf_json = json.dumps(
            {
                "patient_first_name": "John",
                "patient_last_name": "Doe",
                "examination_date": "01.01.2024",
                "casenumber": "12345",
            }
        )

        self.mock_response.json.return_value = {"message": {"content": high_conf_json}}
        self.mock_requests.post.return_value = self.mock_response

        input_text = "Patient John Doe found in examination record on Datum 01.01.2024"

        extractor.extract_metadata_smart_sampling(input_text, confidence_threshold=0.5)

        self.assertEqual(self.mock_requests.post.call_count, 1)


class TestEnrichedMetadataExtractor(unittest.TestCase):
    def setUp(self):
        self.mock_ollama = Mock(spec=OllamaOptimizedExtractor)
        self.mock_optimizer = Mock(spec=FrameSamplingOptimizer)
        self.enriched_extractor = EnrichedMetadataExtractor(
            self.mock_ollama, self.mock_optimizer
        )


class TestVideoMetadataEnricher(unittest.TestCase):
    def setUp(self):
        self.enricher = VideoMetadataEnricher()
        self.enricher.enriched_extractor.extract_from_frame_sequence = Mock(
            return_value={"llm_extracted": {"patient_first_name": "LLM_Name"}}
        )
        # Mock FrameProcessor to just return whatever lists we pass, so we don't need real objects
        self.enricher.frame_processor.process_coroutine_output = Mock(
            side_effect=lambda x: x
        )

    def test_merge_metadata_sources_priority(self):
        legacy_data = {
            "patient_first_name": "OCR_Name_Typos",
            "patient_dob": "01.01.1990",
        }

        # FIX: Pass dummy frame_samples so the enricher actually triggers the extraction logic.
        # If frame_samples is empty, it skips the step and result['enriched_data'] stays empty.
        result = self.enricher.enrich_from_multiple_sources(
            video_path="test.mp4",
            frame_samples=[{"dummy": "data"}],
            ocr_texts=[],
            existing_metadata=legacy_data,
        )

        self.assertEqual(
            result["enriched_data"]["llm_extracted"]["patient_first_name"], "LLM_Name"
        )
        self.assertEqual(result["fallback_data"]["patient_dob"], "01.01.1990")


if __name__ == "__main__":
    unittest.main()
