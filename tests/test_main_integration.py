"""
Tests for main.py integration with Ollama LLM processor.
Tests the complete pipeline integration and compatibility functions.
"""

import pytest
import unittest
from unittest.mock import patch
import sys
import os
from pathlib import Path
import tempfile
import json
from PIL import Image

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../lx_anonymizer"))
)

# Import the modules we want to test
try:
    from lx_anonymizer import main
    from lx_anonymizer.ollama.ollama_llm_processor import (
        OllamaLLMProcessor,
        analyze_full_image_with_ollama,
        replace_phi4_with_ollama,
    )
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)


class TestMainOllamaIntegration(unittest.TestCase):
    """Test main.py integration with Ollama LLM"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock all external dependencies
        self.patches = {}

        # Mock Ollama client
        self.patches["ollama_client"] = patch("lx_anonymizer.ollama_llm.ollama.Client")
        self.mock_ollama_client = self.patches["ollama_client"].start()

        # Mock successful model verification
        self.mock_ollama_client.return_value.list.return_value = {
            "models": [{"name": "llama3.2-vision:latest"}]
        }

        # Mock successful generation
        self.mock_ollama_client.return_value.generate.return_value = {
            "response": json.dumps(
                {
                    "names": [
                        {
                            "full_name": "Test Patient",
                            "first_name": "Test",
                            "last_name": "Patient",
                            "role": "Patient",
                            "confidence": "High",
                        }
                    ]
                }
            )
        }

        # Mock other dependencies
        self.patches["gpu_memory"] = patch("lx_anonymizer.main.clear_gpu_memory")
        self.patches["process_image"] = patch("lx_anonymizer.main.process_image")
        self.patches["get_image_paths"] = patch("lx_anonymizer.main.get_image_paths")
        self.patches["create_results_directory"] = patch(
            "lx_anonymizer.main.create_results_directory"
        )

        # Start all patches
        for patch_name, patch_obj in self.patches.items():
            if patch_name != "ollama_client":  # Already started
                setattr(self, f"mock_{patch_name}", patch_obj.start())

        # Configure mocks
        self.mock_get_image_paths.return_value = [Path("/test/image.jpg")]
        self.mock_create_results_directory.return_value = "/test/results"
        self.mock_process_image.return_value = (
            Path("/test/processed.jpg"),
            {"test": "data"},
        )

    def tearDown(self):
        """Clean up after tests"""
        for patch_obj in self.patches.values():
            patch_obj.stop()

    @patch("lx_anonymizer.main.Path.exists")
    def test_main_with_ollama_integration(self, mock_exists):
        """Test main function with Ollama integration enabled"""
        mock_exists.return_value = True

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            test_image = Image.new("RGB", (100, 100), color="red")
            test_image.save(tmp_file.name)
            test_image_path = tmp_file.name

        try:
            # Test main function call with Ollama integration
            result = main.main(
                image_or_pdf_path=test_image_path,
                device="olympus_cv_1500",
                validation=False,
                disable_llm=False,  # Enable LLM processing
            )

            # Verify the function completed successfully
            self.assertIsNotNone(result)

            # Verify that image processing was called
            self.mock_process_image.assert_called()

        finally:
            # Clean up
            Path(test_image_path).unlink()

    @patch("lx_anonymizer.main.Path.exists")
    def test_main_with_ollama_disabled(self, mock_exists):
        """Test main function with Ollama integration disabled"""
        mock_exists.return_value = True

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            test_image = Image.new("RGB", (100, 100), color="blue")
            test_image.save(tmp_file.name)
            test_image_path = tmp_file.name

        try:
            # Test main function call with Ollama disabled
            result = main.main(
                image_or_pdf_path=test_image_path,
                device="olympus_cv_1500",
                validation=False,
                disable_llm=True,  # Disable LLM processing
            )

            # Verify the function completed successfully
            self.assertIsNotNone(result)

            # Verify that image processing was called
            self.mock_process_image.assert_called()

        finally:
            # Clean up
            Path(test_image_path).unlink()

    @patch("lx_anonymizer.main.read_pdf")
    @patch("lx_anonymizer.main.Path.exists")
    def test_main_with_pdf_and_ollama(self, mock_exists, mock_read_pdf):
        """Test main function with PDF input and Ollama analysis"""
        mock_exists.return_value = True
        mock_read_pdf.return_value = ("Sample PDF text with patient names", True)

        # Create a temporary test PDF (simulate)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"fake pdf content")
            test_pdf_path = tmp_file.name

        try:
            # Test main function call with PDF
            result = main.main(
                image_or_pdf_path=test_pdf_path,
                device="olympus_cv_1500",
                validation=False,
                disable_llm=False,
            )

            # Verify the function completed successfully
            self.assertIsNotNone(result)

            # Verify PDF reading was called
            mock_read_pdf.assert_called_once_with(Path(test_pdf_path))

        finally:
            # Clean up
            Path(test_pdf_path).unlink()


class TestOllamaCompatibilityFunctions(unittest.TestCase):
    """Test compatibility functions for replacing Phi-4 with Ollama"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock Ollama client
        self.mock_client_patcher = patch("lx_anonymizer.ollama_llm.ollama.Client")
        self.mock_client = self.mock_client_patcher.start()

        # Mock successful responses
        self.mock_client.return_value.list.return_value = {
            "models": [{"name": "llama3.2-vision:latest"}]
        }

        self.mock_client.return_value.generate.return_value = {
            "response": json.dumps(
                {
                    "names": [
                        {
                            "full_name": "Dr. Replacement Test",
                            "first_name": "Replacement",
                            "last_name": "Test",
                            "role": "Doctor",
                            "confidence": "High",
                        }
                    ]
                }
            )
        }

    def tearDown(self):
        """Clean up after tests"""
        self.mock_client_patcher.stop()

    def test_phi4_replacement_functionality(self):
        """Test that Ollama can replace Phi-4 functionality"""
        # Create a test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            test_image = Image.new("RGB", (200, 200), color="green")
            test_image.save(tmp_file.name)
            image_path = Path(tmp_file.name)

        try:
            # Test replacement function
            result = replace_phi4_with_ollama(image_path)

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["full_name"], "Dr. Replacement Test")
            self.assertEqual(result[0]["role"], "Doctor")

        finally:
            image_path.unlink()

    @patch("lx_anonymizer.ollama_llm.pytesseract.image_to_string")
    def test_full_image_analysis_compatibility(self, mock_tesseract):
        """Test full image analysis compatibility with existing pipeline"""
        mock_tesseract.return_value = "Medical document with patient information"

        # Create a test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            test_image = Image.new("RGB", (400, 300), color="white")
            test_image.save(tmp_file.name)
            image_path = Path(tmp_file.name)

        # Create temporary CSV directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_dir = Path(tmp_dir)

            try:
                # Test full analysis
                result, csv_path = analyze_full_image_with_ollama(
                    image_path=image_path, csv_dir=csv_dir
                )

                # Verify results
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 1)
                self.assertIsNotNone(csv_path)
                self.assertTrue(csv_path.exists())

                # Verify OCR was called
                mock_tesseract.assert_called_once()

            finally:
                image_path.unlink()


class TestOllamaPerformanceAndMemory(unittest.TestCase):
    """Test performance and memory management aspects"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock Ollama client
        self.mock_client_patcher = patch("lx_anonymizer.ollama_llm.ollama.Client")
        self.mock_client = self.mock_client_patcher.start()

        # Mock successful responses
        self.mock_client.return_value.list.return_value = {
            "models": [{"name": "llama3.2-vision:latest"}]
        }

        self.mock_client.return_value.generate.return_value = {
            "response": json.dumps(
                {
                    "names": [
                        {
                            "full_name": "Performance Test",
                            "role": "Patient",
                            "confidence": "High",
                        }
                    ]
                }
            )
        }

    def tearDown(self):
        """Clean up after tests"""
        self.mock_client_patcher.stop()

    def test_processor_reuse(self):
        """Test that processor can be reused efficiently"""
        processor = OllamaLLMProcessor()

        # Create multiple test images
        test_images = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                test_image = Image.new(
                    "RGB", (100, 100), color=(i * 50, i * 50, i * 50)
                )
                test_image.save(tmp_file.name)
                test_images.append(Path(tmp_file.name))

        try:
            # Process multiple images with same processor
            results = []
            for image_path in test_images:
                result = processor.analyze_image_for_names(image_path)
                results.append(result)

            # Verify all processed successfully
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["full_name"], "Performance Test")

        finally:
            # Clean up
            for image_path in test_images:
                image_path.unlink()

    def test_large_image_handling(self):
        """Test handling of large images"""
        processor = OllamaLLMProcessor()

        # Create a large test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            large_image = Image.new("RGB", (2000, 1500), color="white")
            large_image.save(tmp_file.name)
            large_image_path = Path(tmp_file.name)

        try:
            # Test processing large image
            result = processor.analyze_image_for_names(large_image_path)

            # Verify processing completed
            self.assertIsInstance(result, list)

        finally:
            large_image_path.unlink()


class TestOllamaErrorRecovery(unittest.TestCase):
    """Test error recovery and fallback mechanisms"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock Ollama client
        self.mock_client_patcher = patch("lx_anonymizer.ollama_llm.ollama.Client")
        self.mock_client = self.mock_client_patcher.start()

    def tearDown(self):
        """Clean up after tests"""
        self.mock_client_patcher.stop()

    def test_api_timeout_handling(self):
        """Test handling of API timeouts"""
        # Mock API timeout
        self.mock_client.return_value.list.return_value = {
            "models": [{"name": "llama3.2-vision:latest"}]
        }
        self.mock_client.return_value.generate.side_effect = TimeoutError("API timeout")

        processor = OllamaLLMProcessor()

        # Create test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            test_image = Image.new("RGB", (100, 100), color="red")
            test_image.save(tmp_file.name)
            image_path = Path(tmp_file.name)

        try:
            # Test that timeout is handled gracefully
            result = processor.analyze_image_for_names(image_path)

            # Should return empty list on error
            self.assertEqual(result, [])

        finally:
            image_path.unlink()

    def test_network_error_handling(self):
        """Test handling of network errors"""
        # Mock network error
        self.mock_client.return_value.list.return_value = {
            "models": [{"name": "llama3.2-vision:latest"}]
        }
        self.mock_client.return_value.generate.side_effect = ConnectionError(
            "Network error"
        )

        processor = OllamaLLMProcessor()

        # Test that network error is handled gracefully
        result = processor.analyze_text_for_names("Test text")

        # Should return empty list on error
        self.assertEqual(result, [])


# Test runner
def run_main_integration_tests():
    """Run all main integration tests"""
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestMainOllamaIntegration,
        TestOllamaCompatibilityFunctions,
        TestOllamaPerformanceAndMemory,
        TestOllamaErrorRecovery,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_main_integration_tests()

    # Print summary
    print(f"\n{'=' * 50}")
    print("MAIN INTEGRATION TEST SUMMARY")
    print(f"{'=' * 50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")

    exit(0 if result.wasSuccessful() else 1)
