"""
Comprehensive tests for Ollama LLM integration in lx-anonymizer.
Tests cover initialization, image analysis, text analysis, and integration with main pipeline.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
import json
import tempfile
import csv
from PIL import Image
import numpy as np

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lx_anonymizer')))

# Import the modules we want to test
try:
    from lx_anonymizer.ollama_llm import (
        OllamaLLMProcessor, 
        analyze_full_image_with_ollama,
        replace_phi4_with_ollama,
        initialize_ollama_processor,
        analyze_text_with_ollama
    )
except ImportError as e:
    pytest.skip(f"Could not import ollama_llm module: {e}", allow_module_level=True)


class TestOllamaLLMProcessor(unittest.TestCase):
    """Test cases for OllamaLLMProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_model = "llama3.2-vision:latest"
        self.test_base_url = "http://localhost:11434"
        
        # Create a mock client to avoid actual Ollama calls during testing
        self.mock_client_patcher = patch('lx_anonymizer.ollama_llm.ollama.Client')
        self.mock_client = self.mock_client_patcher.start()
        
        # Mock successful model verification
        self.mock_client.return_value.list.return_value = {
            'models': [{'name': self.test_model}]
        }
        
        # Mock successful generation
        self.mock_client.return_value.generate.return_value = {
            'response': json.dumps({
                'names': [
                    {
                        'full_name': 'Max Mustermann',
                        'first_name': 'Max',
                        'last_name': 'Mustermann',
                        'role': 'Patient',
                        'confidence': 'High',
                        'location': 'top section',
                        'context': 'patient information header'
                    }
                ]
            })
        }
    
    def tearDown(self):
        """Clean up after tests"""
        self.mock_client_patcher.stop()
    
    def test_processor_initialization_success(self):
        """Test successful processor initialization"""
        processor = OllamaLLMProcessor(
            model_name=self.test_model,
            base_url=self.test_base_url
        )
        
        self.assertEqual(processor.model_name, self.test_model)
        self.assertEqual(processor.base_url, self.test_base_url)
        self.assertIsNotNone(processor.client)
    
    def test_processor_initialization_model_not_found(self):
        """Test processor initialization when model is not found"""
        # Mock model not found scenario
        self.mock_client.return_value.list.return_value = {
            'models': [{'name': 'other-model:latest'}]
        }
        
        # Should attempt to pull the model
        processor = OllamaLLMProcessor(
            model_name=self.test_model,
            base_url=self.test_base_url
        )
        
        # Verify pull was called
        self.mock_client.return_value.pull.assert_called_once_with(self.test_model)
    
    def test_encode_image_to_base64(self):
        """Test image encoding to base64"""
        processor = OllamaLLMProcessor()
        
        # Create a test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image = Image.new('RGB', (100, 100), color='red')
            test_image.save(tmp_file.name)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Test encoding
            base64_str = processor._encode_image_to_base64(tmp_path)
            self.assertIsInstance(base64_str, str)
            self.assertGreater(len(base64_str), 0)
        finally:
            # Clean up
            tmp_path.unlink()
    
    def test_analyze_image_for_names_success(self):
        """Test successful image analysis for names"""
        processor = OllamaLLMProcessor()
        
        # Create a test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image = Image.new('RGB', (100, 100), color='blue')
            test_image.save(tmp_file.name)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Test analysis
            result = processor.analyze_image_for_names(tmp_path)
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['full_name'], 'Max Mustermann')
            self.assertEqual(result[0]['role'], 'Patient')
        finally:
            tmp_path.unlink()
    
    def test_analyze_text_for_names_success(self):
        """Test successful text analysis for names"""
        processor = OllamaLLMProcessor()
        
        test_text = "Patient: Max Mustermann, DOB: 1985-01-01, Doctor: Dr. Schmidt"
        
        result = processor.analyze_text_for_names(test_text)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['full_name'], 'Max Mustermann')
    
    def test_validate_and_classify_names(self):
        """Test name validation and classification"""
        processor = OllamaLLMProcessor()
        
        # Mock validation response
        self.mock_client.return_value.generate.return_value = {
            'response': json.dumps({
                'validated_names': [
                    {
                        'original_name': 'Max Mustermann',
                        'corrected_name': 'Max Mustermann',
                        'is_valid': True,
                        'role': 'Patient',
                        'confidence': 'High',
                        'reason': 'Clear patient name format'
                    }
                ]
            })
        }
        
        test_names = ['Max Mustermann', 'Dr. Schmidt']
        result = processor.validate_and_classify_names(test_names)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]['is_valid'])
    
    def test_parse_names_from_response_json(self):
        """Test parsing names from JSON response"""
        processor = OllamaLLMProcessor()
        
        json_response = '''
        Here are the extracted names:
        {
            "names": [
                {
                    "full_name": "Anna Beispiel",
                    "role": "Patient",
                    "confidence": "High"
                }
            ]
        }
        Additional context...
        '''
        
        result = processor._parse_names_from_response(json_response)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['full_name'], 'Anna Beispiel')
    
    def test_parse_names_from_response_text_fallback(self):
        """Test parsing names from structured text when JSON fails"""
        processor = OllamaLLMProcessor()
        
        text_response = '''
        full_name: Peter Mueller
        role: Doctor
        confidence: Medium
        location: signature area
        '''
        
        result = processor._parse_names_from_response(text_response)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['full_name'], 'Peter Mueller')
        self.assertEqual(result[0]['role'], 'Doctor')


class TestOllamaUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_processor_patcher = patch('lx_anonymizer.ollama_llm.OllamaLLMProcessor')
        self.mock_processor_class = self.mock_processor_patcher.start()
        
        # Create mock processor instance
        self.mock_processor = Mock()
        self.mock_processor_class.return_value = self.mock_processor
        
        # Mock successful analysis
        self.mock_processor.analyze_image_for_names.return_value = [
            {
                'full_name': 'Test Patient',
                'first_name': 'Test',
                'last_name': 'Patient',
                'role': 'Patient',
                'confidence': 'High'
            }
        ]
    
    def tearDown(self):
        """Clean up after tests"""
        self.mock_processor_patcher.stop()
    
    @patch('lx_anonymizer.ollama_llm.pytesseract.image_to_string')
    @patch('lx_anonymizer.ollama_llm.Image.open')
    def test_analyze_full_image_with_ollama(self, mock_image_open, mock_tesseract):
        """Test full image analysis with Ollama"""
        # Mock image opening and OCR
        mock_image = Mock()
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value.__enter__.return_value = mock_image
        mock_tesseract.return_value = "Sample OCR text from medical document"
        
        # Create temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image = Image.new('RGB', (200, 200), color='white')
            test_image.save(tmp_file.name)
            image_path = Path(tmp_file.name)
        
        # Create temporary CSV directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_dir = Path(tmp_dir)
            
            try:
                result, csv_path = analyze_full_image_with_ollama(
                    image_path=image_path,
                    csv_dir=csv_dir
                )
                
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]['full_name'], 'Test Patient')
                
                # Check that CSV was created
                self.assertIsNotNone(csv_path)
                self.assertTrue(csv_path.exists())
                
            finally:
                image_path.unlink()
    
    def test_replace_phi4_with_ollama(self):
        """Test replacing Phi-4 analysis with Ollama"""
        # Create temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image = Image.new('RGB', (200, 200), color='green')
            test_image.save(tmp_file.name)
            image_path = Path(tmp_file.name)
        
        try:
            result = replace_phi4_with_ollama(image_path)
            
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['full_name'], 'Test Patient')
            
        finally:
            image_path.unlink()
    
    def test_initialize_ollama_processor(self):
        """Test processor initialization function"""
        processor = initialize_ollama_processor("test-model:latest")
        
        self.assertIsNotNone(processor)
        self.mock_processor_class.assert_called_once_with(model_name="test-model:latest")
    
    def test_analyze_text_with_ollama(self):
        """Test text analysis with Ollama"""
        # Mock text analysis
        self.mock_processor.analyze_text_for_names.return_value = [
            {'full_name': 'Dr. Hans Mueller'},
            {'full_name': 'Patient Schmidt'}
        ]
        
        result = analyze_text_with_ollama("Test medical text with names")
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIn('Dr. Hans Mueller', result)
        self.assertIn('Patient Schmidt', result)


class TestOllamaErrorHandling(unittest.TestCase):
    """Test error handling in Ollama integration"""
    
    def setUp(self):
        """Set up test fixtures for error scenarios"""
        self.mock_client_patcher = patch('lx_anonymizer.ollama_llm.ollama.Client')
        self.mock_client = self.mock_client_patcher.start()
    
    def tearDown(self):
        """Clean up after tests"""
        self.mock_client_patcher.stop()
    
    def test_processor_initialization_failure(self):
        """Test processor initialization failure"""
        # Mock initialization failure
        self.mock_client.side_effect = Exception("Connection failed")
        
        with self.assertRaises(Exception):
            OllamaLLMProcessor()
    
    def test_image_analysis_with_invalid_image(self):
        """Test image analysis with invalid image path"""
        # Mock successful initialization
        self.mock_client.return_value.list.return_value = {
            'models': [{'name': 'llama3.2-vision:latest'}]
        }
        
        processor = OllamaLLMProcessor()
        
        # Test with non-existent image
        invalid_path = Path("/nonexistent/image.jpg")
        result = processor.analyze_image_for_names(invalid_path)
        
        self.assertEqual(result, [])
    
    def test_api_call_failure(self):
        """Test handling of API call failures"""
        # Mock successful initialization but failed generation
        self.mock_client.return_value.list.return_value = {
            'models': [{'name': 'llama3.2-vision:latest'}]
        }
        self.mock_client.return_value.generate.side_effect = Exception("API Error")
        
        processor = OllamaLLMProcessor()
        
        # Create test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image = Image.new('RGB', (100, 100), color='red')
            test_image.save(tmp_file.name)
            tmp_path = Path(tmp_file.name)
        
        try:
            result = processor.analyze_image_for_names(tmp_path)
            self.assertEqual(result, [])
        finally:
            tmp_path.unlink()


class TestOllamaIntegrationWithRealData(unittest.TestCase):
    """Integration tests with more realistic data"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.mock_client_patcher = patch('lx_anonymizer.ollama_llm.ollama.Client')
        self.mock_client = self.mock_client_patcher.start()
        
        # Mock more realistic responses
        self.mock_client.return_value.list.return_value = {
            'models': [{'name': 'llama3.2-vision:latest'}]
        }
    
    def tearDown(self):
        """Clean up after tests"""
        self.mock_client_patcher.stop()
    
    def test_medical_document_analysis(self):
        """Test analysis of medical document with multiple names"""
        # Mock response with multiple names
        self.mock_client.return_value.generate.return_value = {
            'response': json.dumps({
                'names': [
                    {
                        'full_name': 'Maria Schneider',
                        'first_name': 'Maria',
                        'last_name': 'Schneider',
                        'role': 'Patient',
                        'confidence': 'High',
                        'location': 'patient header',
                        'context': 'patient information section'
                    },
                    {
                        'full_name': 'Dr. Thomas Weber',
                        'first_name': 'Thomas',
                        'last_name': 'Weber',
                        'role': 'Doctor',
                        'confidence': 'High',
                        'location': 'signature area',
                        'context': 'physician signature'
                    },
                    {
                        'full_name': 'Schwester Anna',
                        'first_name': 'Anna',
                        'last_name': '',
                        'role': 'Nurse',
                        'confidence': 'Medium',
                        'location': 'notes section',
                        'context': 'nursing notes'
                    }
                ]
            })
        }
        
        processor = OllamaLLMProcessor()
        
        # Create test image representing medical document
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Create a more complex test image
            test_image = Image.new('RGB', (800, 600), color='white')
            test_image.save(tmp_file.name)
            tmp_path = Path(tmp_file.name)
        
        try:
            result = processor.analyze_image_for_names(tmp_path)
            
            self.assertEqual(len(result), 3)
            
            # Check patient
            patient = next(r for r in result if r['role'] == 'Patient')
            self.assertEqual(patient['full_name'], 'Maria Schneider')
            self.assertEqual(patient['confidence'], 'High')
            
            # Check doctor
            doctor = next(r for r in result if r['role'] == 'Doctor')
            self.assertEqual(doctor['full_name'], 'Dr. Thomas Weber')
            
            # Check nurse
            nurse = next(r for r in result if r['role'] == 'Nurse')
            self.assertEqual(nurse['full_name'], 'Schwester Anna')
            self.assertEqual(nurse['confidence'], 'Medium')
            
        finally:
            tmp_path.unlink()
    
    def test_csv_output_format(self):
        """Test CSV output format and content"""
        # Mock response
        self.mock_client.return_value.generate.return_value = {
            'response': json.dumps({
                'names': [
                    {
                        'full_name': 'Test Patient',
                        'first_name': 'Test',
                        'last_name': 'Patient',
                        'role': 'Patient',
                        'confidence': 'High',
                        'location': 'header',
                        'context': 'patient info'
                    }
                ]
            })
        }
        
        # Create test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            test_image = Image.new('RGB', (200, 200), color='blue')
            test_image.save(tmp_file.name)
            image_path = Path(tmp_file.name)
        
        # Create temporary directory for CSV
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_dir = Path(tmp_dir)
            
            try:
                with patch('lx_anonymizer.ollama_llm.pytesseract.image_to_string') as mock_ocr:
                    mock_ocr.return_value = "Sample medical text"
                    
                    result, csv_path = analyze_full_image_with_ollama(
                        image_path=image_path,
                        csv_dir=csv_dir
                    )
                
                # Verify CSV was created and has correct format
                self.assertIsNotNone(csv_path)
                self.assertTrue(csv_path.exists())
                
                # Read and verify CSV content
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]['full_name'], 'Test Patient')
                    self.assertEqual(rows[0]['role'], 'Patient')
                    self.assertEqual(rows[0]['confidence'], 'High')
                
            finally:
                image_path.unlink()


# Test suite setup
def create_test_suite():
    """Create comprehensive test suite for Ollama integration"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestOllamaLLMProcessor,
        TestOllamaUtilityFunctions,
        TestOllamaErrorHandling,
        TestOllamaIntegrationWithRealData
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


if __name__ == '__main__':
    # Run the comprehensive test suite
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"OLLAMA INTEGRATION TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)