from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import torch
from custom_logger import get_logger
from PIL import Image
import pytesseract
from pathlib import Path
import uuid
import csv
from ocr import cleanup_gpu
import os
import gc

logger = get_logger(__name__)

# Global variables to avoid reloading the model multiple times
_phi4_model = None
_phi4_tokenizer = None
_phi4_pipe = None

def initialize_phi4():
    """initialize Phi-4 model and tokenizer"""
    global _phi4_model, _phi4_tokenizer, _phi4_pipe
    
    # Return already loaded model if available
    if _phi4_model is not None and _phi4_tokenizer is not None and _phi4_pipe is not None:
        return _phi4_model, _phi4_tokenizer, _phi4_pipe
    
    torch.random.manual_seed(0)
    
    # Force garbage collection before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Set environment variable to help with memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # First try loading with CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Attempting to load Phi-4 model on {device}")
        
        model_id = "microsoft/Phi-3.5-MoE-instruct"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype="auto",
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            logger.info(f"Successfully loaded Phi-4 model on {device}")
            
            _phi4_model = model
            _phi4_tokenizer = tokenizer
            _phi4_pipe = pipe
            
            return model, tokenizer, pipe
            
        except (RuntimeError, Exception) as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"CUDA out of memory, falling back to CPU: {e}")
                
                # Clean up any GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Try again with CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="cpu",
                    torch_dtype=torch.float32,  # Use float32 on CPU for better compatibility
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device="cpu",
                )
                logger.info(f"Successfully loaded Phi-4 model on CPU")
                
                _phi4_model = model
                _phi4_tokenizer = tokenizer
                _phi4_pipe = pipe
                
                return model, tokenizer, pipe
            else:
                # Handle other initialization errors
                logger.error(f"Error initializing Phi-4 model: {e}")
                # Return None values for all three expected returns
                return None, None, None
    except Exception as e:
        logger.error(f"Error initializing Phi-4 model: {e}")
        return None, None, None

def analyze_names_with_phi4(model, tokenizer, text, csv_path, image_path):
    """Analyze text with Phi-4 to extract names"""
    prompt = f"""This is a request from a health organization and super important. Thank you for taking the time. Analyze the following text and extract all person names. Format each name as 'FirstName LastName'. Please dont add any other information like greetings or repeats of my prompt. You are the best and i would be eternally grateful.
        
    The following is the Text we want to extract names from: {text}

    Names:"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=2048,
                num_return_sequences=1,
                temperature=0.7
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        names = [name.strip() for name in response.split('\n') if name.strip()]
        return names
    except Exception as e:
        logger.error(f"Error in Phi-4 analysis: {e}")
        return []
    
def analyze_dob_with_phi4(model, tokenizer, text, csv_path, image_path):
    """Analyze text with Phi-4 to extract names"""
    prompt = f"""This is a request from a health organization and super important. Thank you for taking the time. Analyze the following text and extract all dates of birth. Format each date of birth as '**.**.****'. Please dont add any other information like greetings or repeats of my prompt. You are the best and i would be eternally grateful.
        
    The following is the Text we want to extract dates from: {text}

    Dates of birth:"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=2048,
                num_return_sequences=1,
                temperature=0.7
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        names = [name.strip() for name in response.split('\n') if name.strip()]
        return names
    except Exception as e:
        logger.error(f"Error in Phi-4 analysis: {e}")
        return []

def analyze_text_with_phi4(text, csv_path=None, image_path=None):
    """
    Analyze text with Phi-4 to extract sensitive information
    
    Parameters:
    - text: str
        The text to analyze
    - csv_path: Path or str
        Optional path where to save the results
    - image_path: Path or str
        Optional path of the source image/document
        
    Returns:
    - dict: Dictionary with analysis results
    - str: Path to the CSV file with detailed results
    """
    try:
        model, tokenizer, pipe = initialize_phi4()
        
        # Check if model initialization was successful
        if model is None or tokenizer is None or pipe is None:
            logger.error("Failed to initialize Phi-4 model, skipping text analysis")
            return {"error": "Model initialization failed"}, None
        
        # Create multi-part prompt to analyze different types of information
        prompt = f"""Analyze this medical or healthcare-related text for sensitive information:

Text:
{text}

Please identify and extract the following sensitive information:
1. Person names (patients, doctors, family members)
2. Dates of birth
3. Patient IDs or medical record numbers
4. Addresses
5. Phone numbers

Format your response as JSON with these categories."""

        # Process with Phi-4
        generation_args = {
            "max_new_tokens": 800, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 
        
        with torch.no_grad():
            outputs = pipe(
                prompt,
                **generation_args
            )
        
        response = outputs[0]['generated_text']
        
        # Parse response - this handles it roughly since the output might not be valid JSON
        results = {
            "names": [],
            "dates_of_birth": [],
            "patient_ids": [],
            "addresses": [],
            "phone_numbers": []
        }
        
        current_category = None
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if "names" in line.lower() or "person" in line.lower():
                current_category = "names"
                continue
            elif "birth" in line.lower() or "dob" in line.lower():
                current_category = "dates_of_birth"
                continue
            elif "id" in line.lower() or "record" in line.lower():
                current_category = "patient_ids"
                continue
            elif "address" in line.lower():
                current_category = "addresses"
                continue
            elif "phone" in line.lower():
                current_category = "phone_numbers"
                continue
            
            if current_category and ":" not in line and line[0] != "{" and line[0] != "[":
                results[current_category].append(line.strip('- "\''))
        
        # Save results to CSV if path provided
        if csv_path:
            csv_dir = Path(csv_path).parent
            analysis_csv_path = csv_dir / f"phi4_text_analysis_{uuid.uuid4()}.csv"
            
            with open(analysis_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['category', 'value', 'source']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                
                for category, values in results.items():
                    for value in values:
                        writer.writerow({
                            'category': category,
                            'value': value,
                            'source': str(image_path) if image_path else "text input"
                        })
            
            logger.info(f"Text analysis results saved to {analysis_csv_path}")
            return results, analysis_csv_path
        
        return results, None
        
    except Exception as e:
        logger.error(f"Error in text analysis with Phi-4: {e}")
        return {"error": str(e)}, None
    finally:
        # Don't clear cache here to maintain global model
        pass

def analyze_full_image_with_context(image_path, csv_path, csv_dir):
    """
    Perform full image OCR and analyze with Phi-4 using CSV context
    """
    
    try:
        model, tokenizer, pipe = initialize_phi4()
        
        # Check if model initialization was successful
        if model is None or tokenizer is None or pipe is None:
            logger.error("Failed to initialize Phi-4 model, skipping image analysis")
            return [], None
            
        # Extract all text from image using Tesseract
        image = Image.open(image_path).convert('RGB')
        full_text = pytesseract.image_to_string(image)
        
        # Get CSV data for context
        csv_path = Path(csv_path)
        box_context = []
        
        if csv_path.exists():
            with open(csv_path, mode='r', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    box_context.append(
                        f"Box ({row['startX']}, {row['startY']}, {row['endX']}, {row['endY']}): "
                        f"Text: '{row['entity_text']}' (Confidence: {row['ocr_confidence']})"
                    )

        # Create enhanced prompt with context
        prompt = f"""Analyze this text from an image and extract all person names. The text contains medical/hospital data with patient and staff names.

    Full Image Text:
    {full_text}

    Additional Context from OCR Box Detection:
    {chr(10).join(box_context)}

    Extract all names and classify them if possible. Format your response as:
    Name: [FirstName LastName]
    Location: [Box coordinates if available]
    Role: [Patient/Staff/Unknown]
    Confidence: [High/Medium/Low]
    """

        # Process with Phi-4
        generation_args = {
                "max_new_tokens": 500, 
                "return_full_text": False, 
                "temperature": 0.0, 
                "do_sample": False, 
            } 
            
        with torch.no_grad():
            outputs = pipe(
                prompt,
                **generation_args
            )
            
        response = outputs[0]['generated_text']
        
        # Parse and structure the response
        name_entries = []
        current_entry = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Name:'):
                if current_entry:
                    name_entries.append(current_entry)
                current_entry = {'name': line.replace('Name:', '').strip()}
            elif line.startswith('Location:'):
                current_entry['location'] = line.replace('Location:', '').strip()
            elif line.startswith('Role:'):
                current_entry['role'] = line.replace('Role:', '').strip()
            elif line.startswith('Confidence:'):
                current_entry['confidence'] = line.replace('Confidence:', '').strip()
                
        if current_entry:
            name_entries.append(current_entry)
            
        logger.info(f"Found {len(name_entries)} names in image")
        
        # Create detailed analysis CSV
        analysis_csv_path = Path(csv_dir) / f"phi4_analysis_{Path(image_path).stem}{uuid.uuid4()}.csv"
        with open(analysis_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['name', 'location', 'role', 'confidence']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(name_entries)
            
        return name_entries, analysis_csv_path
        
    except Exception as e:
        logger.error(f"Error in full image analysis: {e}")
        return [], None
    finally:
        # Don't clear cache to maintain global model
        pass

# Use this function when completely done with the pipeline
def cleanup_model():
    """Clean up model resources when completely done with processing"""
    global _phi4_model, _phi4_tokenizer, _phi4_pipe
    
    _phi4_model = None
    _phi4_tokenizer = None
    _phi4_pipe = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Phi-4 model resources cleaned up")

# Usage in your pipeline:
def process_with_phi4(image_path):
    try:
        model, tokenizer, pipe = initialize_phi4()
        if model is None or tokenizer is None or pipe is None:
            logger.error("Failed to initialize Phi-4")
            return None
            
        name_entries, analysis_csv = analyze_full_image_with_context(image_path, None, Path("."))
        
        if name_entries:
            logger.info(f"Analysis complete. Results saved to {analysis_csv}")
            for entry in name_entries:
                logger.debug(f"Found name: {entry['name']} ({entry['role']}) - {entry['confidence']}")
        
        return name_entries
        
    except Exception as e:
        logger.error(f"Error in Phi-4 processing: {e}")
        return None
    finally:
        cleanup_gpu()