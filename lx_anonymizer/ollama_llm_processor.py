"""
Ollama-based LLM integration for image processing and text analysis.
This module replaces the Phi-4 based LLM functions with Ollama-based implementations.
"""

import json
import logging
import csv
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import base64
import io
from PIL import Image
import pytesseract
import ollama
from .custom_logger import get_logger

logger = get_logger(__name__)


class OllamaLLMProcessor:
    """
    Ollama-based LLM processor for image analysis and text processing.
    """
    
    def __init__(self, model_name: str = "llama3.2-vision:latest", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama LLM processor.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        
        # Verify model availability
        try:
            self._verify_model_availability()
            logger.info(f"Ollama LLM processor initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM processor: {e}")
            raise
        
    def call_llm(self, prompt: str, context=None,
             temperature: float = 0.1, top_p: float = 0.9,
             image: Optional[Path] = None) -> dict:

        kwargs = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": 1000
            },
            "stream": False          # optional: disable streaming for one-shot
        }

        if context:
            kwargs["context"] = context          # keeps chat memory
        if image:
            kwargs["images"] = [self._encode_image_to_base64(image)]

        return self.client.generate(**kwargs)

    
    def _verify_model_availability(self) -> bool:
        """
        Verify that the specified model is available in Ollama.
        
        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                # Try to pull the model
                logger.info(f"Attempting to pull model: {self.model_name}")
                self.client.pull(self.model_name)
                logger.info(f"Successfully pulled model: {self.model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying model availability: {e}")
            return False
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        """
        Encode image to base64 string for Ollama API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (optional optimization)
                max_size = (1024, 1024)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                return img_str
                
        except Exception as e:
            logger.error(f"Error encoding image to base64: {e}")
            return ""
    
    def analyze_image_for_names(self, image_path: Path, context_data: Optional[Dict] = None) -> List[Dict]:
        """
        Analyze image for person names using Ollama vision model.
        
        Args:
            image_path: Path to the image file
            context_data: Optional context data from OCR or other sources
            
        Returns:
            List[Dict]: List of detected names with metadata
        """
        try:
            # Encode image
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                logger.error("Failed to encode image")
                return []
            
            # Prepare context information
            context_str = ""
            if context_data:
                context_str = f"\n\nAdditional context:\n{json.dumps(context_data, indent=2)}"
            
            # Create prompt for name detection
            prompt = f"""
            Analyze this medical document image and extract ALL person names (both patients and medical staff).
            
            Instructions:
            1. Look for names in headers, signatures, patient information sections, and doctor names
            2. Extract names in the format "FirstName LastName"
            3. Classify each name as Patient, Doctor, Nurse, or Unknown
            4. Provide confidence level (High/Medium/Low)
            5. If possible, provide approximate location coordinates in the image
            
            Return the results in JSON format:
            {{
                "names": [
                    {{
                        "full_name": "FirstName LastName",
                        "first_name": "FirstName",
                        "last_name": "LastName",
                        "role": "Patient|Doctor|Nurse|Unknown",
                        "confidence": "High|Medium|Low",
                        "location": "description of where found in image",
                        "context": "surrounding text or context"
                    }}
                ]
            }}
            {context_str}
            """
            
            # Make API call to Ollama
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_base64],
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            )
            
            # Parse response
            response_text = response['response']
            logger.debug(f"Ollama response: {response_text}")
            
            # Extract JSON from response
            names_data = self._parse_names_from_response(response_text)
            
            logger.info(f"Extracted {len(names_data)} names from image using Ollama")
            return names_data
            
        except Exception as e:
            logger.error(f"Error analyzing image with Ollama: {e}")
            return []
    
    def _parse_names_from_response(self, response_text: str) -> List[Dict]:
        """
        Parse names from Ollama response text.
        
        Args:
            response_text: Raw response text from Ollama
            
        Returns:
            List[Dict]: Parsed names data
        """
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
                return data.get('names', [])
            
            # Fallback: parse structured text
            names = []
            lines = response_text.split('\n')
            current_name = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('full_name:') or line.startswith('Full Name:'):
                    if current_name:
                        names.append(current_name)
                    current_name = {'full_name': line.split(':', 1)[1].strip()}
                elif line.startswith('role:') or line.startswith('Role:'):
                    current_name['role'] = line.split(':', 1)[1].strip()
                elif line.startswith('confidence:') or line.startswith('Confidence:'):
                    current_name['confidence'] = line.split(':', 1)[1].strip()
                elif line.startswith('location:') or line.startswith('Location:'):
                    current_name['location'] = line.split(':', 1)[1].strip()
            
            if current_name:
                names.append(current_name)
            
            return names
            
        except Exception as e:
            logger.error(f"Error parsing names from response: {e}")
            return []
    
    def analyze_text_for_names(self, text: str, context: Optional[str] = None) -> List[Dict]:
        """
        Analyze text for person names using Ollama text model.
        
        Args:
            text: Text to analyze
            context: Optional context information
            
        Returns:
            List[Dict]: List of detected names with metadata
        """
        try:
            context_str = f"\n\nContext: {context}" if context else ""
            
            prompt = f"""
            Analyze the following text from a medical document and extract all person names.
            
            Text to analyze:
            {text}
            {context_str}
            
            Instructions:
            1. Extract all person names (patients, doctors, nurses, staff)
            2. Classify each name by likely role
            3. Provide confidence level
            4. Return in JSON format
            
            JSON format:
            {{
                "names": [
                    {{
                        "full_name": "FirstName LastName",
                        "first_name": "FirstName",
                        "last_name": "LastName",
                        "role": "Patient|Doctor|Nurse|Staff|Unknown",
                        "confidence": "High|Medium|Low",
                        "context": "surrounding text"
                    }}
                ]
            }}
            """
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 800
                }
            )
            
            response_text = response['response']
            names_data = self._parse_names_from_response(response_text)
            
            logger.info(f"Extracted {len(names_data)} names from text using Ollama")
            return names_data
            
        except Exception as e:
            logger.error(f"Error analyzing text with Ollama: {e}")
            return []
    
    def validate_and_classify_names(self, names: List[str], context: Optional[str] = None) -> List[Dict]:
        """
        Validate and classify a list of names using Ollama.
        
        Args:
            names: List of names to validate
            context: Optional context information
            
        Returns:
            List[Dict]: Validated and classified names
        """
        try:
            names_str = "\n".join([f"- {name}" for name in names])
            context_str = f"\n\nContext: {context}" if context else ""
            
            prompt = f"""
            Validate and classify the following list of names from a medical document.
            
            Names to validate:
            {names_str}
            {context_str}
            
            For each name, determine:
            1. Is it a valid person name?
            2. What is the likely role (Patient, Doctor, Nurse, Staff)?
            3. Confidence level in the classification
            4. Any corrections needed to the name format
            
            Return in JSON format:
            {{
                "validated_names": [
                    {{
                        "original_name": "original name",
                        "corrected_name": "corrected name if needed",
                        "is_valid": true/false,
                        "role": "Patient|Doctor|Nurse|Staff|Unknown",
                        "confidence": "High|Medium|Low",
                        "reason": "explanation for classification"
                    }}
                ]
            }}
            """
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            )
            
            response_text = response['response']
            
            # Parse validation results
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    data = json.loads(json_str)
                    return data.get('validated_names', [])
            except:
                pass
            
            # Fallback validation
            validated = []
            for name in names:
                validated.append({
                    'original_name': name,
                    'corrected_name': name,
                    'is_valid': True,
                    'role': 'Unknown',
                    'confidence': 'Medium',
                    'reason': 'Fallback validation'
                })
            
            return validated
            
        except Exception as e:
            logger.error(f"Error validating names with Ollama: {e}")
            return []


def analyze_full_image_with_ollama(image_path: Path, csv_path: Optional[Path] = None, 
                                  csv_dir: Optional[Path] = None) -> Tuple[List[Dict], Optional[Path]]:
    """
    Perform full image analysis using Ollama LLM with context from OCR results.
    
    Args:
        image_path: Path to the image file
        csv_path: Optional path to existing CSV with OCR results
        csv_dir: Directory to save analysis results
        
    Returns:
        Tuple[List[Dict], Optional[Path]]: Detected names and path to analysis CSV
    """
    try:
        # Initialize Ollama processor
        processor = OllamaLLMProcessor()
        
        # Load context data from CSV if available
        context_data = {}
        if csv_path and csv_path.exists():
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    context_data['ocr_results'] = list(reader)
            except Exception as e:
                logger.warning(f"Could not load CSV context: {e}")
        
        # Extract full text using Tesseract for additional context
        try:
            image = Image.open(image_path).convert('RGB')
            full_text = pytesseract.image_to_string(image)
            context_data['full_text'] = full_text
        except Exception as e:
            logger.warning(f"Could not extract full text: {e}")
        
        # Analyze image with Ollama
        name_entries = processor.analyze_image_for_names(image_path, context_data)
        
        # Save results to CSV if directory provided
        analysis_csv_path = None
        if csv_dir and name_entries:
            csv_dir = Path(csv_dir)
            csv_dir.mkdir(parents=True, exist_ok=True)
            
            analysis_csv_path = csv_dir / f"ollama_analysis_{image_path.stem}_{uuid.uuid4().hex[:8]}.csv"
            
            with open(analysis_csv_path, 'w', newline='', encoding='utf-8') as f:
                if name_entries:
                    fieldnames = name_entries[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(name_entries)
            
            logger.info(f"Ollama analysis results saved to: {analysis_csv_path}")
        
        return name_entries, analysis_csv_path
        
    except Exception as e:
        logger.error(f"Error in full image analysis with Ollama: {e}")
        return [], None

# Compatibility functions to replace existing Phi-4 functions

def replace_phi4_with_ollama(image_path: Path, existing_csv_path: Optional[Path] = None) -> Optional[List[Dict]]:
    """
    Replace Phi-4 analysis with Ollama-based analysis.
    
    Args:
        image_path: Path to the image file
        existing_csv_path: Optional path to existing analysis CSV
        
    Returns:
        Optional[List[Dict]]: Analysis results or None if failed
    """
    try:
        logger.info(f"Starting Ollama-based analysis for: {image_path}")
        
        # Determine CSV directory
        csv_dir = image_path.parent / "ollama_analysis"
        
        # Perform analysis
        name_entries, analysis_csv = analyze_full_image_with_ollama(
            image_path=image_path,
            csv_path=existing_csv_path,
            csv_dir=csv_dir
        )
        
        if name_entries:
            logger.info(f"Ollama analysis complete. Found {len(name_entries)} names")
            for entry in name_entries:
                logger.debug(f"Found name: {entry.get('full_name', 'Unknown')} "
                           f"({entry.get('role', 'Unknown')}) - {entry.get('confidence', 'Unknown')}")
        else:
            logger.warning("No names found in Ollama analysis")
        
        return name_entries
        
    except Exception as e:
        logger.error(f"Error in Ollama processing: {e}")
        return None


def initialize_ollama_processor(model_name: str = "llama3.2-vision:latest") -> Optional[OllamaLLMProcessor]:
    """
    Initialize Ollama processor (replacement for initialize_phi4).
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        Optional[OllamaLLMProcessor]: Processor instance or None if failed
    """
    try:
        return OllamaLLMProcessor(model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Ollama processor: {e}")
        return None


def analyze_text_with_ollama(text: str, csv_path: Optional[Path] = None, 
                           image_path: Optional[Path] = None) -> List[str]:
    """
    Analyze text with Ollama (replacement for analyze_text_with_phi4).
    
    Args:
        text: Text to analyze
        csv_path: Optional path to CSV file for context
        image_path: Optional path to image file for context
        
    Returns:
        List[str]: List of extracted names
    """
    try:
        processor = OllamaLLMProcessor()
        
        # Load context from CSV if available
        context = ""
        if csv_path and csv_path.exists():
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    context = f"OCR results context: {f.read()}"
            except Exception as e:
                logger.warning(f"Could not load CSV context: {e}")
        
        # Analyze text
        name_entries = processor.analyze_text_for_names(text, context)
        
        # Extract just the names for compatibility
        names = [entry.get('full_name', '') for entry in name_entries if entry.get('full_name')]
        
        return names
        
    except Exception as e:
        logger.error(f"Error analyzing text with Ollama: {e}")
        return []