from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import torch
from .custom_logger import get_logger
from PIL import Image
import pytesseract
from pathlib import Path
import uuid
import csv
from .ocr import cleanup_gpu

logger = get_logger(__name__)


def initialize_phi4():
    """initialize Phi-4 model and tokenizer"""
    torch.random.manual_seed(0)
    try:
        model = AutoModelForCausalLM.from_pretrained( 
            "microsoft/Phi-3.5-MoE-instruct",  
            device_map="cuda",  
            torch_dtype="auto",  
            trust_remote_code=False,  
        )         
        model = AutoModelForCausalLM.from_pretrained("matteogeniaccio/phi-4", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct") 
        pipe = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
        ) 
        if torch.cuda.is_available():
            model = model.cuda()
        return model, tokenizer, pipe
    except Exception as e:
        logger.error(f"Error initializing Phi-4 model: {e}")
        return None, None

def analyze_text_with_phi4(model, tokenizer, text, csv_path, image_path):
    
    """Analyze text with Phi-4 to extract names"""
    prompt = f"""Analyze the following text and extract all person names. Format each name as 'FirstName LastName'.
        
    Text: {text}

    Names:"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
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
    
def analyze_full_image_with_context(image_path, csv_path, csv_dir):
    """
    Perform full image OCR and analyze with Phi-4 using CSV context
    """
    
    try:
        model, tokenizer, pipe = initialize_phi4()
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
        inputs = prompt
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        generation_args = {
                "max_new_tokens": 500, 
                "return_full_text": False, 
                "temperature": 0.0, 
                "do_sample": False, 
            } 
            
            
        with torch.no_grad():
            outputs = pipe(
                inputs,
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Usage in your pipeline:
def process_with_phi4(image_path):
    try:
        model, tokenizer = initialize_phi4()
        if model is None or tokenizer is None:
            logger.error("Failed to initialize Phi-4")
            return None
            
        name_entries, analysis_csv = analyze_full_image_with_context(image_path, model, tokenizer)
        
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