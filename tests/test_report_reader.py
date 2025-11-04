"""
Comprehensive tests for the ReportReader module.

This test suite covers:
- ReportReader initialization and configuration
- PDF text extraction functionality
- Metadata extraction with different methods (SpaCy, LLM)
- Text anonymization functionality
- Error handling and edge cases
- OCR fallback mechanisms
- Ollama/LLM integration
"""

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the modules we're testing
try:
    from lx_anonymizer.report_reader import ReportReader
    from lx_anonymizer.settings import DEFAULT_SETTINGS
except ImportError:
    pytest.skip("lx_anonymizer modules not available", allow_module_level=True)


class TestReportReader:
    """Test cases for ReportReader functionality."""

    @pytest.fixture
    def test_text(self):
        """Sample German medical report text for testing."""
        return """
        ENDOSKOPIEBERICHT
        
        Patient: Max Mustermann
        Geburtsdatum: 15.06.1985
        Fallnummer: 12345
        Geschlecht: männlich
        
        Untersuchungsdatum: 20.03.2024
        Untersuchungszeit: 14:30
        Untersuchender Arzt: Dr. Schmidt
        
        BEFUND:
        Es wurde eine Magenspiegelung durchgeführt.
        Der Befund war unauffällig.
        
        Zentrum: Universitätsklinikum München
        """

    @pytest.fixture
    def sample_metadata(self):
        """Expected metadata extraction results."""
        return {
            "patient_first_name": "Max",
            "patient_last_name": "Mustermann",
            "patient_dob": "15.06.1985",
            "casenumber": "12345",
            "patient_gender": "männlich",
            "examination_date": "20.03.2024",
            "examination_time": "14:30",
            "examiner_first_name": "Dr.",
            "examiner_last_name": "Schmidt",
            "center": "Universitätsklinikum München",
        }

    @pytest.fixture
    def mock_pdf_file(self):
        """Create a temporary mock PDF file."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            # Create minimal PDF-like content for testing
            tmp_file.write(b"%PDF-1.4\n%mock pdf content for testing\n%%EOF")
            tmp_file.flush()
            yield Path(tmp_file.name)
            # Cleanup
            os.unlink(tmp_file.name)

    def test_report_reader_initialization_default(self):
        """Test ReportReader initialization with default settings."""
        reader = ReportReader()

        assert reader.locale == DEFAULT_SETTINGS["locale"]
        assert reader.text_date_format == DEFAULT_SETTINGS["text_date_format"]
        assert reader.employee_first_names == DEFAULT_SETTINGS["first_names"]
        assert reader.employee_last_names == DEFAULT_SETTINGS["last_names"]
        assert reader.flags == DEFAULT_SETTINGS["flags"]

        # Check extractors are initialized
        assert hasattr(reader, "patient_extractor")
        assert hasattr(reader, "examiner_extractor")
        assert hasattr(reader, "endoscope_extractor")
        assert hasattr(reader, "examination_extractor")

    def test_report_reader_initialization_custom(self):
        """Test ReportReader initialization with custom parameters."""
        custom_first_names = ["Johannes", "Maria"]
        custom_last_names = ["Müller", "Weber"]
        custom_locale = "en_US"
        custom_date_format = "%Y-%m-%d"

        reader = ReportReader(
            locale=custom_locale,
            employee_first_names=custom_first_names,
            employee_last_names=custom_last_names,
            text_date_format=custom_date_format,
        )

        assert reader.locale == custom_locale
        assert reader.text_date_format == custom_date_format
        assert reader.employee_first_names == custom_first_names
        assert reader.employee_last_names == custom_last_names

    @patch("lx_anonymizer.report_reader.pdfplumber")
    def test_read_pdf_success(self, mock_pdfplumber, mock_pdf_file, test_text):
        """Test successful PDF text extraction."""
        # Mock pdfplumber behavior
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = test_text
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        reader = ReportReader()
        result = reader.read_pdf(str(mock_pdf_file))

        assert result == test_text
        mock_pdfplumber.open.assert_called_once_with(str(mock_pdf_file))

    def test_read_pdf_nonexistent_file(self):
        """Test PDF reading with non-existent file."""
        reader = ReportReader()
        result = reader.read_pdf("/path/that/does/not/exist.pdf")

        assert result == ""

    def test_read_pdf_invalid_input(self):
        """Test PDF reading with invalid input types."""
        reader = ReportReader()

        # Test None input
        result = reader.read_pdf(None)
        assert result == ""

        # Test invalid type
        result = reader.read_pdf(123)
        assert result == ""

    def test_extract_report_meta_spacy(self, test_text):
        """Test metadata extraction using SpaCy extractors."""
        reader = ReportReader()

        # Mock the extractors to return expected results
        with patch.object(reader, "extract_report_meta_deepseek", return_value={}):
            with patch.object(
                reader.patient_extractor,
                "__call__",
                return_value={
                    "patient_first_name": "Max",
                    "patient_last_name": "Mustermann",
                    "patient_dob": "15.06.1985",
                    "casenumber": "12345",
                    "patient_gender": "männlich",
                },
            ):
                result = reader.extract_report_meta(test_text, None)

        assert "patient_first_name" in result
        assert "patient_last_name" in result
        assert "patient_dob" in result
        assert "casenumber" in result
        assert "patient_gender" in result

    def test_extract_report_meta_empty_text(self):
        """Test metadata extraction with empty text."""
        reader = ReportReader()
        result = reader.extract_report_meta("", None)

        # Should return basic structure even with empty text
        assert isinstance(result, dict)

    def test_anonymize_report(self, test_text, sample_metadata):
        """Test text anonymization functionality."""
        reader = ReportReader()

        anonymized = reader.anonymize_report(test_text, sample_metadata)

        # Check that sensitive information is replaced
        assert "Max Mustermann" not in anonymized
        assert "12345" not in anonymized  # Case number should be anonymized
        assert "15.06.1985" not in anonymized  # DOB should be anonymized

        # Check that structure is preserved
        assert "ENDOSKOPIEBERICHT" in anonymized
        assert "BEFUND" in anonymized

    def test_pdf_hash(self, mock_pdf_file):
        """Test PDF hash calculation."""
        reader = ReportReader()

        # Read the mock PDF file
        with open(mock_pdf_file, "rb") as f:
            pdf_content = f.read()

        result = reader.pdf_hash(pdf_content)
        expected_hash = hashlib.sha256(pdf_content).hexdigest()

        assert result == expected_hash
        assert len(result) == 64  # SHA256 hex digest length

    @patch("lx_anonymizer.report_reader.pdfplumber")
    def test_process_report_full_pipeline(
        self, mock_pdfplumber, mock_pdf_file, test_text
    ):
        """Test the complete process_report pipeline."""
        # Mock PDF reading
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = test_text
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        reader = ReportReader()

        # Mock metadata extraction to avoid LLM dependency
        with patch.object(reader, "extract_report_meta_deepseek", return_value={}):
            with patch.object(
                reader.patient_extractor,
                "__call__",
                return_value={
                    "patient_first_name": "Max",
                    "patient_last_name": "Mustermann",
                },
            ):
                original_text, anonymized_text, metadata, pdf_path = (
                    reader.process_report(
                        pdf_path=str(mock_pdf_file),
                        use_llm_extractor="spacy",  # Use string instead of None
                    )
                )

        assert original_text == test_text
        assert (
            anonymized_text != original_text
        )  # Should be different after anonymization
        assert isinstance(metadata, dict)
        assert pdf_path is None  # No anonymized PDF requested

    def test_process_report_with_text_input(self, test_text):
        """Test process_report when text is provided directly."""
        reader = ReportReader()

        with patch.object(reader, "extract_report_meta_deepseek", return_value={}):
            with patch.object(reader.patient_extractor, "__call__", return_value={}):
                original_text, anonymized_text, metadata, pdf_path = (
                    reader.process_report(
                        text=test_text,
                        use_llm_extractor="spacy",  # Use string instead of None
                    )
                )

        assert original_text == test_text
        assert isinstance(anonymized_text, str)
        assert isinstance(metadata, dict)

    def test_process_report_invalid_inputs(self):
        """Test process_report with invalid inputs."""
        reader = ReportReader()

        # No inputs provided
        with pytest.raises(
            ValueError,
            match="Either 'pdf_path' 'image_path' or 'text' must be provided",
        ):
            reader.process_report()

    def test_process_report_missing_file(self):
        """Test process_report with missing PDF file."""
        reader = ReportReader()

        original_text, anonymized_text, metadata, pdf_path = reader.process_report(
            pdf_path="/nonexistent/file.pdf"
        )

        # Should return empty strings and dict when file doesn't exist
        assert original_text == ""
        assert anonymized_text == ""
        assert metadata == {}

    @patch("lx_anonymizer.report_reader.OllamaOptimizedExtractor")
    def test_ollama_integration_available(self, mock_ollama_extractor):
        """Test ReportReader when Ollama is available."""
        # Mock successful Ollama initialization
        mock_extractor_instance = Mock()
        mock_extractor_instance.current_model = "deepseek-r1"
        mock_ollama_extractor.return_value = mock_extractor_instance

        with patch("lx_anonymizer.report_reader.ensure_ollama", return_value=Mock()):
            reader = ReportReader()

        assert hasattr(reader, "ollama_available")
        assert hasattr(reader, "ollama_extractor")

    @patch("lx_anonymizer.report_reader.ensure_ollama")
    def test_ollama_integration_unavailable(self, mock_ensure_ollama):
        """Test ReportReader when Ollama is unavailable."""
        # Mock Ollama failure
        mock_ensure_ollama.side_effect = Exception("Ollama not available")

        reader = ReportReader()

        assert hasattr(reader, "ollama_available")
        assert hasattr(reader, "ollama_extractor")

    def test_extract_report_meta_deepseek_unavailable(self, test_text):
        """Test DeepSeek extraction when Ollama is unavailable."""
        reader = ReportReader()
        # Mock unavailable state
        with patch.object(reader, "ollama_available", False):
            with patch.object(reader, "ollama_extractor", None):
                result = reader.extract_report_meta_deepseek(test_text)

        assert result == {}

    @patch("lx_anonymizer.report_reader.tesseract_full_image_ocr")
    @patch("lx_anonymizer.report_reader.Image")
    def test_process_report_with_image(self, mock_image, mock_ocr):
        """Test process_report with image input."""
        # Mock image processing
        mock_pil_image = Mock()
        mock_image.open.return_value = mock_pil_image
        mock_ocr.return_value = ("extracted text", 0.95)

        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_file.flush()

            try:
                reader = ReportReader()

                with patch.object(
                    reader, "extract_report_meta_deepseek", return_value={}
                ):
                    with patch.object(
                        reader.patient_extractor, "__call__", return_value={}
                    ):
                        original_text, anonymized_text, metadata, pdf_path = (
                            reader.process_report(
                                image_path=tmp_file.name, use_llm_extractor="spacy"
                            )
                        )

                assert original_text == "extracted text"
                mock_image.open.assert_called_once_with(tmp_file.name)
                mock_ocr.assert_called_once()

            finally:
                os.unlink(tmp_file.name)

    def test_process_report_insufficient_text(self, mock_pdf_file):
        """Test process_report when extracted text is too short."""
        reader = ReportReader()

        # Mock very short text extraction
        with patch.object(reader, "read_pdf", return_value="hi"):
            original_text, anonymized_text, metadata, pdf_path = reader.process_report(
                pdf_path=str(mock_pdf_file)
            )

        # Should skip metadata extraction for insufficient text
        assert original_text == "hi"
        assert isinstance(metadata, dict)
        # Metadata should be minimal when text is insufficient
        assert (
            len([k for k, v in metadata.items() if v is not None]) <= 1
        )  # Only hash if any


class TestReportReaderMetadataExtraction:
    """Focused tests for metadata extraction methods."""

    @pytest.fixture
    def reader_with_mocked_llm(self):
        """ReportReader with mocked LLM extraction."""
        reader = ReportReader()
        # Mock available state using properties
        reader.ollama_available = True
        reader.ollama_extractor = Mock()
        return reader

    def test_extract_report_meta_deepseek_success(self, reader_with_mocked_llm):
        """Test successful DeepSeek metadata extraction."""
        # Mock successful extraction
        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "patient_first_name": "Max",
            "patient_last_name": "Mustermann",
        }
        reader_with_mocked_llm.ollama_extractor.extract_metadata.return_value = (
            mock_result
        )

        result = reader_with_mocked_llm.extract_report_meta_deepseek("test text")

        assert result["patient_first_name"] == "Max"
        assert result["patient_last_name"] == "Mustermann"

    def test_extract_report_meta_deepseek_failure(self, reader_with_mocked_llm):
        """Test DeepSeek extraction failure handling."""
        # Mock extraction failure
        reader_with_mocked_llm.ollama_extractor.extract_metadata.side_effect = (
            Exception("Model error")
        )

        result = reader_with_mocked_llm.extract_report_meta_deepseek("test text")

        assert result == {}

    def test_extract_report_meta_deepseek_empty_result(self, reader_with_mocked_llm):
        """Test DeepSeek extraction with empty result."""
        # Mock empty extraction
        reader_with_mocked_llm.ollama_extractor.extract_metadata.return_value = None

        result = reader_with_mocked_llm.extract_report_meta_deepseek("test text")

        assert result == {}


class TestReportReaderErrorHandling:
    """Tests for error handling and edge cases."""

    def test_process_report_ocr_fallback(self, mock_pdf_file):
        """Test OCR fallback when PDF extraction fails."""
        reader = ReportReader()

        # Mock PDF reading to return insufficient text
        with patch.object(reader, "read_pdf", return_value=""):
            with patch(
                "lx_anonymizer.report_reader.convert_pdf_to_images"
            ) as mock_convert:
                with patch(
                    "lx_anonymizer.report_reader.tesseract_full_image_ocr"
                ) as mock_ocr:
                    # Mock image conversion and OCR
                    mock_convert.return_value = [Mock()]  # Mock PIL image
                    mock_ocr.return_value = ("OCR extracted text", 0.9)

                    original_text, anonymized_text, metadata, pdf_path = (
                        reader.process_report(pdf_path=str(mock_pdf_file))
                    )

                    # Should have used OCR fallback
                    mock_convert.assert_called_once()
                    mock_ocr.assert_called_once()

    def test_process_report_ocr_fallback_failure(self, mock_pdf_file):
        """Test when both PDF extraction and OCR fallback fail."""
        reader = ReportReader()

        with patch.object(reader, "read_pdf", return_value=""):
            with patch(
                "lx_anonymizer.report_reader.convert_pdf_to_images",
                side_effect=Exception("OCR failed"),
            ):
                original_text, anonymized_text, metadata, pdf_path = (
                    reader.process_report(pdf_path=str(mock_pdf_file))
                )

                # Should return original (empty) text when all methods fail
                assert original_text == ""
                assert anonymized_text == ""
                assert metadata == {}

    def test_anonymization_with_empty_metadata(self):
        """Test anonymization when metadata extraction returns empty results."""
        reader = ReportReader()

        test_text = "Patient: John Doe"
        empty_metadata = {}

        result = reader.anonymize_report(test_text, empty_metadata)

        # Should still process even with empty metadata
        assert isinstance(result, str)
        # Text might not be significantly changed without metadata, but shouldn't crash


class TestReportReaderIntegration:
    """Integration tests combining multiple components."""

    @patch("lx_anonymizer.report_reader.pdfplumber")
    def test_full_pipeline_with_llm_extraction(self, mock_pdfplumber, mock_pdf_file):
        """Test complete pipeline with LLM extraction enabled."""
        # Mock PDF reading
        test_text = "Patient: Max Mustermann, DOB: 15.06.1985"
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = test_text
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        reader = ReportReader()

        # Mock LLM extraction
        with patch.object(
            reader,
            "extract_report_meta_deepseek",
            return_value={
                "patient_first_name": "Max",
                "patient_last_name": "Mustermann",
                "patient_dob": "15.06.1985",
            },
        ):
            original_text, anonymized_text, metadata, pdf_path = reader.process_report(
                pdf_path=str(mock_pdf_file), use_llm_extractor="deepseek"
            )

        assert original_text == test_text
        assert metadata["patient_first_name"] == "Max"
        assert metadata["patient_last_name"] == "Mustermann"
        assert "Max Mustermann" not in anonymized_text  # Should be anonymized

    def test_batch_processing_simulation(self):
        """Simulate batch processing scenario."""
        reader = ReportReader()

        test_texts = [
            "Patient: Alice Smith, DOB: 01.01.1990",
            "Patient: Bob Johnson, DOB: 02.02.1985",
            "Patient: Carol White, DOB: 03.03.1980",
        ]

        results = []
        for text in test_texts:
            with patch.object(
                reader.patient_extractor,
                "__call__",
                return_value={
                    "patient_first_name": text.split()[1],
                    "patient_last_name": text.split()[2].rstrip(","),
                },
            ):
                original, anonymized, metadata, _ = reader.process_report(text=text)
                results.append((original, anonymized, metadata))

        assert len(results) == 3
        for original, anonymized, metadata in results:
            assert original != anonymized  # Should be different after anonymization
            assert isinstance(metadata, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
