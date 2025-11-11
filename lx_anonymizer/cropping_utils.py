"""
Utility-Funktionen für das Sensitive Region Cropping.

Diese Datei enthält Hilfsfunktionen für erweiterte Cropping-Operationen
und Batch-Verarbeitung von PDFs.
"""

import json
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .report_reader import ReportReader
from .custom_logger import get_logger

logger = get_logger(__name__)


class BatchCropper:
    """
    Klasse für die Batch-Verarbeitung mehrerer PDFs mit Cropping-Funktionalität.
    """
    
    def __init__(self, 
                 output_base_dir: str,
                 max_workers: int = 4,
                 locale: str = 'de_DE'):
        """
        Initialisiert den BatchCropper.
        
        Args:
            output_base_dir: Basis-Ausgabeverzeichnis
            max_workers: Maximale Anzahl paralleler Threads
            locale: Lokalisierung für die Textverarbeitung
        """
        self.output_base_dir = Path(output_base_dir)
        self.max_workers = max_workers
        self.locale = locale
        
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisiere ReportReader
        self.reader = ReportReader(locale=locale)
        
        # Statistiken
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'total_regions': 0,
            'processing_time': 0.0
        }

    def process_pdf_directory(self, 
                            pdf_dir: str, 
                            file_pattern: str = "*.pdf",
                            create_report: bool = True) -> Dict[str, Any]:
        """
        Verarbeitet alle PDFs in einem Verzeichnis.
        
        Args:
            pdf_dir: Verzeichnis mit PDFs
            file_pattern: Dateimuster für PDF-Dateien
            create_report: Ob ein Zusammenfassungs-Report erstellt werden soll
            
        Returns:
            Dictionary mit Verarbeitungsstatistiken und Ergebnissen
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            raise ValueError(f"PDF-Verzeichnis existiert nicht: {pdf_dir}")
        
        # Finde alle PDF-Dateien
        pdf_files = list(pdf_dir.glob(file_pattern))
        if not pdf_files:
            logger.warning(f"Keine PDF-Dateien gefunden in: {pdf_dir} mit Muster: {file_pattern}")
            return {'files': [], 'stats': self.stats}
        
        logger.info(f"Gefunden {len(pdf_files)} PDF-Dateien zur Verarbeitung")
        
        start_time = time.time()
        results = []
        
        # Verarbeite Dateien parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Starte alle Tasks
            future_to_pdf = {
                executor.submit(self._process_single_pdf, pdf_file): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Sammle Ergebnisse
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        self.stats['successful'] += 1
                        self.stats['total_regions'] += result.get('total_regions', 0)
                    else:
                        self.stats['failed'] += 1
                    
                    self.stats['processed'] += 1
                    
                    logger.info(f"Verarbeitet ({self.stats['processed']}/{len(pdf_files)}): {pdf_file.name}")
                    
                except Exception as e:
                    logger.error(f"Fehler bei Verarbeitung von {pdf_file.name}: {e}")
                    results.append({
                        'pdf_path': str(pdf_file),
                        'success': False,
                        'error': str(e),
                        'total_regions': 0
                    })
                    self.stats['failed'] += 1
                    self.stats['processed'] += 1
        
        self.stats['processing_time'] = time.time() - start_time
        
        # Erstelle Zusammenfassungs-Report
        batch_result = {
            'files': results,
            'stats': self.stats.copy(),
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_directory': str(pdf_dir),
            'output_directory': str(self.output_base_dir)
        }
        
        if create_report:
            report_path = self._create_batch_report(batch_result)
            batch_result['report_path'] = report_path
        
        logger.info(f"Batch-Verarbeitung abgeschlossen: {self.stats['successful']}/{len(pdf_files)} erfolgreich")
        
        return batch_result

    def _process_single_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Verarbeitet eine einzelne PDF-Datei.
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            
        Returns:
            Dictionary mit Verarbeitungsergebnis
        """
        try:
            # Erstelle Ausgabeverzeichnis für diese PDF
            pdf_output_dir = self.output_base_dir / pdf_path.stem
            
            # Verwende die erweiterte process_report_with_cropping Methode
            original_text, anonymized_text, report_meta, cropped_regions = self.reader.process_report_with_cropping(
                pdf_path=str(pdf_path),
                crop_output_dir=str(pdf_output_dir),
                crop_sensitive_regions=True,
                use_llm_extractor='deepseek',
                verbose=False
            )
            
            # Sammle Statistiken
            total_regions = report_meta.get('total_cropped_regions', 0)
            
            result = {
                'pdf_path': str(pdf_path),
                'success': True,
                'output_dir': str(pdf_output_dir),
                'total_regions': total_regions,
                'cropped_regions': cropped_regions,
                'metadata': {
                    'patient_name': f"{report_meta.get('patient_first_name', '')} {report_meta.get('patient_last_name', '')}".strip(),
                    'case_number': report_meta.get('casenumber'),
                    'patient_dob': str(report_meta.get('patient_dob', '')),
                    'pdf_hash': report_meta.get('pdf_hash'),
                    'text_length': len(original_text) if original_text else 0
                },
                'cropping_enabled': report_meta.get('cropping_enabled', False)
            }
            
            # Speichere Metadaten als JSON
            meta_file = pdf_output_dir / f"{pdf_path.stem}_metadata.json"
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(str(meta_file), 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei Verarbeitung von {pdf_path}: {e}")
            return {
                'pdf_path': str(pdf_path),
                'success': False,
                'error': str(e),
                'total_regions': 0
            }

    def _create_batch_report(self, batch_result: Dict[str, Any]) -> str:
        """
        Erstellt einen detaillierten Batch-Report.
        
        Args:
            batch_result: Ergebnisse der Batch-Verarbeitung
            
        Returns:
            Pfad zur Report-Datei
        """
        report_path = self.output_base_dir / "batch_cropping_report.json"
        
        with open(str(report_path), 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, indent=2, ensure_ascii=False, default=str)
        
        # Erstelle auch einen Human-Readable Report
        txt_report_path = self.output_base_dir / "batch_cropping_report.txt"
        self._create_human_readable_report(batch_result, txt_report_path)
        
        logger.info(f"Batch-Report erstellt: {report_path}")
        return str(report_path)

    def _create_human_readable_report(self, batch_result: Dict[str, Any], output_path: Path):
        """
        Erstellt einen menschenlesbaren Batch-Report.
        """
        stats = batch_result['stats']
        
        with open(str(output_path), 'w', encoding='utf-8') as f:
            f.write("=== BATCH CROPPING REPORT ===\n\n")
            f.write(f"Verarbeitungsdatum: {batch_result['processing_date']}\n")
            f.write(f"Eingabeverzeichnis: {batch_result['input_directory']}\n")
            f.write(f"Ausgabeverzeichnis: {batch_result['output_directory']}\n\n")
            
            f.write("STATISTIKEN:\n")
            f.write(f"  Gesamt verarbeitet: {stats['processed']}\n")
            f.write(f"  Erfolgreich: {stats['successful']}\n")
            f.write(f"  Fehlgeschlagen: {stats['failed']}\n")
            f.write(f"  Gefundene Regionen: {stats['total_regions']}\n")
            f.write(f"  Verarbeitungszeit: {stats['processing_time']:.2f} Sekunden\n\n")
            
            f.write("DETAILLIERTE ERGEBNISSE:\n")
            f.write("-" * 50 + "\n")
            
            for result in batch_result['files']:
                f.write(f"\nDatei: {Path(result['pdf_path']).name}\n")
                f.write(f"  Status: {'✅ Erfolgreich' if result['success'] else '❌ Fehlgeschlagen'}\n")
                
                if result['success']:
                    f.write(f"  Gefundene Regionen: {result['total_regions']}\n")
                    if result.get('metadata'):
                        meta = result['metadata']
                        if meta.get('patient_name'):
                            f.write(f"  Patient: {meta['patient_name']}\n")
                        if meta.get('case_number'):
                            f.write(f"  Fallnummer: {meta['case_number']}\n")
                        f.write(f"  Textlänge: {meta.get('text_length', 0)} Zeichen\n")
                else:
                    f.write(f"  Fehler: {result.get('error', 'Unbekannter Fehler')}\n")


def create_cropping_config(output_file: str = "cropping_config.json") -> str:
    """
    Erstellt eine Beispiel-Konfigurationsdatei für das Cropping.
    
    Args:
        output_file: Pfad zur Ausgabe-Konfigurationsdatei
        
    Returns:
        Pfad zur erstellten Konfigurationsdatei
    """
    config = {
        "cropping_settings": {
            "margin": 20,
            "min_region_size": [100, 30],
            "merge_distance": 50
        },
        "sensitive_patterns": {
            "patient_name": r"[A-ZÄÖÜ][a-zäöüß]+\s*,\s*[A-ZÄÖÜ][a-zäöüß]+",
            "birth_date": r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
            "case_number": r"(?:Fallnummer|Fallnr|Fall\.Nr)[:\s]*(\d+)",
            "social_security": r"\b\d{2}\s?\d{2}\s?\d{2}\s?\d{4}\b",
            "phone_number": r"\b(?:\+49\s?)?(?:\d{3,5}[\s\-]?)?\d{6,8}\b",
            "address": r"[A-ZÄÖÜ][a-zäöüß\s]+(str\.|straße|platz|weg|gasse)\s*\d+",
            "doctor_name": r"(?:Dr\.\s?(?:med\.\s?)?)?[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?"
        },
        "processing_options": {
            "use_llm_extractor": "deepseek",
            "use_ensemble_ocr": False,
            "create_visualizations": True,
            "parallel_processing": True,
            "max_workers": 4
        },
        "output_settings": {
            "create_metadata_files": True,
            "create_batch_reports": True,
            "image_format": "PNG",
            "image_quality": 95
        }
    }
    
    with open(str(output_file), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Konfigurationsdatei erstellt: {output_file}")
    return output_file


def validate_cropping_results(output_dir: str) -> Dict[str, Any]:
    """
    Validiert die Ergebnisse einer Cropping-Operation.
    
    Args:
        output_dir: Verzeichnis mit Cropping-Ergebnissen
        
    Returns:
        Validierungs-Report
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        return {'valid': False, 'error': f'Ausgabeverzeichnis existiert nicht: {output_dir}'}
    
    # Sammle alle Crop-Bilder
    crop_images = list(output_dir.glob("**/*.png")) + list(output_dir.glob("**/*.jpg"))
    metadata_files = list(output_dir.glob("**/*_metadata.json"))
    
    validation_result = {
        'valid': True,
        'output_directory': str(output_dir),
        'total_crop_images': len(crop_images),
        'total_metadata_files': len(metadata_files),
        'validation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'issues': []
    }
    
    # Validiere Metadaten-Dateien
    for meta_file in metadata_files:
        try:
            with open(str(meta_file), 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
                
            # Prüfe ob die referenzierten Crop-Bilder existieren
            if 'cropped_regions' in meta_data:
                for page, crop_files in meta_data['cropped_regions'].items():
                    for crop_file in crop_files:
                        if not Path(crop_file).exists():
                            validation_result['issues'].append(
                                f"Crop-Bild nicht gefunden: {crop_file} (referenziert in {meta_file.name})"
                            )
                            
        except Exception as e:
            validation_result['issues'].append(f"Fehler beim Lesen von {meta_file.name}: {e}")
    
    # Prüfe auf verwaiste Crop-Bilder
    referenced_crops = set()
    for meta_file in metadata_files:
        try:
            with open(str(meta_file), 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            if 'cropped_regions' in meta_data:
                for crop_files in meta_data['cropped_regions'].values():
                    referenced_crops.update(crop_files)
        except Exception:
            continue
    
    orphaned_images = []
    for crop_image in crop_images:
        if str(crop_image) not in referenced_crops:
            orphaned_images.append(str(crop_image))
    
    if orphaned_images:
        validation_result['orphaned_images'] = orphaned_images
        validation_result['issues'].append(f"Gefunden {len(orphaned_images)} verwaiste Crop-Bilder")
    
    # Setze Validitätsstatus
    validation_result['valid'] = len(validation_result['issues']) == 0
    
    return validation_result
