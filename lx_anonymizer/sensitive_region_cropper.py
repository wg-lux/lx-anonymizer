"""
Sensitive Region Cropper - Implementierung für das Cropping von Regionen mit sensitivem Text.

Dieses Modul erkennt sensitive Textregionen in PDFs und erstellt gecropte Versionen
der entsprechenden Bereiche für die Anonymisierung.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw

from .custom_logger import get_logger
from .ocr import tesseract_full_image_ocr
from .pdf_operations import convert_pdf_to_images
from .spacy_extractor import ExaminerDataExtractor, PatientDataExtractor

logger = get_logger(__name__)


class SensitiveRegionCropper:
    """
    Klasse für das Erkennen und Cropping von sensitiven Textregionen in medizinischen Dokumenten.
    """

    def __init__(
        self,
        margin: int = 20,
        min_region_size: Tuple[int, int] = (100, 30),
        merge_distance: int = 50,
    ):
        """
        Initialisiert den SensitiveRegionCropper.

        Args:
            margin: Zusätzlicher Rand um sensitive Bereiche in Pixeln
            min_region_size: Minimale Größe für Crop-Regionen (Breite, Höhe)
            merge_distance: Maximaler Abstand zum Zusammenführen benachbarter Regionen
        """
        self.margin = margin
        self.min_region_size = min_region_size
        self.merge_distance = merge_distance

        # Initialisiere Extraktoren für sensitive Daten
        self.patient_extractor = PatientDataExtractor()
        self.examiner_extractor = ExaminerDataExtractor()

        self.patient_info = None

        # Definiere sensitive Datentypen und ihre Regex-Patterns
        self.sensitive_patterns = {
            "patient_first_name": r"[A-ZÄÖÜ][a-zäöüß]+\s*,\s*[A-ZÄÖÜ][a-zäöüß]+",
            "patient_last_name": r"[A-ZÄÖÜ][a-zäöüß]+\s*,\s*[A-ZÄÖÜ][a-zäöüß]+",
            "birth_date": r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
            "casenumber": r"(?:Fallnummer|Fallnr|Fall\.Nr)[:\s]*(\d+)",
            "social_security": r"\b\d{2}\s?\d{2}\s?\d{2}\s?\d{4}\b",
            "phone_number": r"\b(?:\+49\s?)?(?:\d{3,5}[\s\-]?)?\d{6,8}\b",
            "address": r"[A-ZÄÖÜ][a-zäöüß\s]+(str\.|straße|platz|weg|gasse)\s*\d+",
            "examiner_first_name": r"(?:Dr\.\s?(?:med\.\s?)?)?[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?",
            "examiner_last_name": r"(?:Dr\.\s?(?:med\.\s?)?)?[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?"
        }

    # Füge in der Klasse SensitiveRegionCropper hinzu:

    def _enclosing_box(
        self, boxes: Iterable[Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        """
        Enclosing-Box-Helfer: (x, y, w, h)-Boxen -> (x1, y1, x2, y2).
        """
        boxes = list(boxes)
        if not boxes:
            return (0, 0, 0, 0)
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        return (x1, y1, x2, y2)

    def _create_bounding_boxes_recursive(
        self,
        boxes: List[Tuple[int, int, int, int]],
        axis: str = "x",
        gap_threshold: Optional[int] = None,
        depth: int = 0,
        max_depth: int = 10,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Rekursive Zerlegung einer Menge von (x, y, w, h)-Boxen in kleinere
        Enclosing-Boxes, indem entlang einer Achse bei ausreichend großen Lücken
        gesplittet wird. Achse wechselt pro Rekursionsschritt (x -> y -> x ...).

        Args:
            boxes: Liste von (x, y, w, h).
            axis: 'x' (horizontal clustern) oder 'y' (vertikal clustern).
            gap_threshold: minimale Lücke (in Pixeln), um zu splitten.
            depth: aktuelle Rekursionstiefe.
            max_depth: Sicherheitslimit gegen Pathologien.

        Returns:
            Liste von (x1, y1, x2, y2)-Bounding-Boxes.
        """
        if not boxes:
            return []
        if len(boxes) == 1 or depth >= max_depth:
            return [self._enclosing_box(boxes)]

        if gap_threshold is None:
            # konservativ: halbiere merge_distance, aber min. 8 px
            gap_threshold = max(8, self.merge_distance // 2)

        # Sortiere Boxen entlang Achse und berechne Lücken
        if axis == "x":
            boxes_sorted = sorted(boxes, key=lambda b: (b[0], b[1]))
            # gap = linker Rand des nächsten - rechter Rand des vorherigen
            gaps = []
            for i in range(len(boxes_sorted) - 1):
                left = boxes_sorted[i]
                right = boxes_sorted[i + 1]
                gap = (right[0]) - (left[0] + left[2])
                gaps.append((gap, i))
        else:  # axis == "y"
            boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
            # gap = oberer Rand des nächsten - unterer Rand des vorherigen
            gaps = []
            for i in range(len(boxes_sorted) - 1):
                top = boxes_sorted[i]
                bottom = boxes_sorted[i + 1]
                gap = (bottom[1]) - (top[1] + top[3])
                gaps.append((gap, i))

        # Finde größte Lücke über Schwellwert
        split_index = None
        max_gap = -1
        for gap, i in gaps:
            if gap > gap_threshold and gap > max_gap:
                max_gap = gap
                split_index = i

        if split_index is not None:
            # Splitte in zwei Cluster und rekursiv weiter
            left_cluster = boxes_sorted[: split_index + 1]
            right_cluster = boxes_sorted[split_index + 1 :]

            next_axis = "y" if axis == "x" else "x"

            left_boxes = self._create_bounding_boxes_recursive(
                left_cluster,
                axis=next_axis,
                gap_threshold=gap_threshold,
                depth=depth + 1,
                max_depth=max_depth,
            )
            right_boxes = self._create_bounding_boxes_recursive(
                right_cluster,
                axis=next_axis,
                gap_threshold=gap_threshold,
                depth=depth + 1,
                max_depth=max_depth,
            )
            return left_boxes + right_boxes

        # Keine ausreichend große Lücke auf dieser Achse -> Versuche einmal die andere Achse
        if depth == 0:
            other_axis = "y" if axis == "x" else "x"
            alt = self._create_bounding_boxes_recursive(
                boxes,
                axis=other_axis,
                gap_threshold=gap_threshold,
                depth=depth + 1,
                max_depth=max_depth,
            )
            return alt

        # Keine sinnvolle weitere Zerlegung -> eine Enclosing-Box
        return [self._enclosing_box(boxes)]

    def detect_sensitive_regions(
        self,
        image: Image.Image,
        word_boxes: List[Tuple[str, Tuple[int, int, int, int]]],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Erkennt sensitive Textregionen basierend auf OCR-Ergebnissen und Regex-Patterns.

        Args:
            image: PIL Image Objekt
            word_boxes: Liste von (word, (x, y, width, height)) Tupeln aus OCR

        Returns:
            Liste von (x1, y1, x2, y2) Bounding Boxes für sensitive Regionen
        """
        sensitive_regions = []
        full_text = " ".join([word for word, _ in word_boxes])

        logger.info(f"Analysiere Text auf sensitive Daten: {full_text[:100]}...")

        # 1. Verwende SpaCy-Extraktoren für strukturierte Erkennung
        patient_info = self.patient_extractor(full_text)

        # 2. Finde Positionen von sensitiven Daten in den Word-Boxes
        for data_type, pattern in self.sensitive_patterns.items():
            matches = re.finditer(pattern, full_text, re.IGNORECASE)

            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                matched_text = match.group()

                logger.debug(
                    f"Gefunden {data_type}: '{matched_text}' an Position {start_pos}-{end_pos}"
                )

                # Finde entsprechende Word-Boxes für diesen Text
                region_boxes = self._find_word_boxes_for_text_span(
                    matched_text, word_boxes, start_pos, end_pos
                )

                if region_boxes:
                    # Erstelle Bounding Boxes für diese Region
                    bboxes = self._create_bounding_boxes_recursive(
                        region_boxes, axis="x"
                    )
                    sensitive_regions.extend(bboxes)  # extend statt append!

        # 3. Zusätzliche Erkennung für Patientendaten aus SpaCy-Extraktor
        if patient_info:
            sensitive_regions.extend(
                self._find_patient_data_regions(patient_info, word_boxes, full_text)
            )

        # 4. Führe benachbarte Regionen zusammen
        logger.debug(f"Sensitive regions vor Merge: {len(sensitive_regions)} Regionen")
        for i, region in enumerate(sensitive_regions):
            if not isinstance(region, tuple) or len(region) != 4:
                logger.error(
                    f"Invalid region at index {i}: {region} (type: {type(region)})"
                )

        # Filtere ungültige Regionen heraus
        valid_regions = [
            r for r in sensitive_regions if isinstance(r, tuple) and len(r) == 4
        ]
        if len(valid_regions) != len(sensitive_regions):
            logger.warning(
                f"Filtered out {len(sensitive_regions) - len(valid_regions)} invalid regions"
            )

        # merged_regions = self._merge_nearby_regions(valid_regions)
        merged_regions = valid_regions  # Deaktiviere Merging für präzisere Crops

        # 5. Erweitere Regionen um Margin und validiere Größe
        final_regions = []
        for x1, y1, x2, y2 in merged_regions:
            x1 = max(0, x1 - self.margin)
            y1 = max(0, y1 - self.margin)
            x2 = min(image.width, x2 + self.margin)
            y2 = min(image.height, y2 + self.margin)

            # Überprüfe minimale Größe
            if (
                x2 - x1 >= self.min_region_size[0]
                and y2 - y1 >= self.min_region_size[1]
            ):
                final_regions.append((x1, y1, x2, y2))

        logger.info(f"Gefunden {len(final_regions)} sensitive Regionen zum Cropping")
        return final_regions

    def _find_word_boxes_for_text_span(
        self,
        target_text: str,
        word_boxes: List[Tuple[str, Tuple[int, int, int, int]]],
        start_pos: int = 0,
        end_pos: int = 0,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Findet Word-Boxes, die einem bestimmten Text entsprechen.
        """
        matching_boxes = []
        target_words = target_text.lower().split()

        # Einfacher Ansatz: Suche nach aufeinanderfolgenden Words
        for i, (word, bbox) in enumerate(word_boxes):
            if word.lower() in target_words:
                matching_boxes.append(bbox)

                # Versuche, aufeinanderfolgende passende Words zu finden
                j = i + 1
                word_idx = target_words.index(word.lower()) + 1

                while (
                    j < len(word_boxes)
                    and word_idx < len(target_words)
                    and word_boxes[j][0].lower() == target_words[word_idx]
                ):
                    matching_boxes.append(word_boxes[j][1])
                    j += 1
                    word_idx += 1

                if (
                    len(matching_boxes) >= len(target_words) / 2
                ):  # Mindestens 50% der Words gefunden
                    break

        return matching_boxes

    def _find_patient_data_regions(
        self,
        patient_info: Dict,
        word_boxes: List[Tuple[str, Tuple[int, int, int, int]]],
        full_text: str,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Findet Regionen basierend auf extrahierten Patientendaten.
        """
        regions = []

        if patient_info.get("patient_first_name") and patient_info.get(
            "patient_last_name"
        ):
            full_name = f"{patient_info['patient_first_name']} {patient_info['patient_last_name']}"
            name_boxes = self._find_word_boxes_for_text_span(full_name, word_boxes)
            if name_boxes:
                # ALT:
                # regions.append(self._create_bounding_box(name_boxes))
                regions.extend(
                    self._create_bounding_boxes_recursive(name_boxes, axis="x")
                )

        # Geburtsdatum
        if patient_info.get("patient_dob"):
            dob_str = str(patient_info["patient_dob"])
            dob_boxes = self._find_word_boxes_for_text_span(dob_str, word_boxes)
            if dob_boxes:
                regions.extend(
                    self._create_bounding_boxes_recursive(dob_boxes, axis="x")
                )

        # Fallnummer
        if patient_info.get("casenumber"):
            case_str = str(patient_info["casenumber"])
            case_boxes = self._find_word_boxes_for_text_span(case_str, word_boxes)
            if case_boxes:
                regions.extend(
                    self._create_bounding_boxes_recursive(case_boxes, axis="x")
                )

        return regions

    def _create_bounding_box(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        """
        Erstellt eine umschließende Bounding Box aus mehreren kleineren Boxes.
        """
        if not boxes:
            return (0, 0, 0, 0)

        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[0] + box[2] for box in boxes)  # x + width
        y2 = max(box[1] + box[3] for box in boxes)  # y + height

        return (x1, y1, x2, y2)

    def _merge_nearby_regions(
        self, regions: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Führt benachbarte Regionen zusammen, die näher als merge_distance sind.
        """
        if not regions:
            return []

        merged = []
        regions = sorted(regions)  # Sortiere nach x1-Koordinate

        current_region = regions[0]

        for next_region in regions[1:]:
            # Prüfe, ob Regionen nah genug sind, um sie zusammenzuführen
            if self._regions_should_merge(current_region, next_region):
                # Führe Regionen zusammen
                current_region = (
                    min(current_region[0], next_region[0]),
                    min(current_region[1], next_region[1]),
                    max(current_region[2], next_region[2]),
                    max(current_region[3], next_region[3]),
                )
            else:
                merged.append(current_region)
                current_region = next_region

        merged.append(current_region)
        return merged

    def _regions_should_merge(
        self, region1: Tuple[int, int, int, int], region2: Tuple[int, int, int, int]
    ) -> bool:
        """
        Bestimmt, ob zwei Regionen zusammengeführt werden sollten.
        """
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2

        # Berechne Abstand zwischen Regionen
        horizontal_distance = max(0, max(x1_1, x1_2) - min(x2_1, x2_2))
        vertical_distance = max(0, max(y1_1, y1_2) - min(y2_1, y2_2))

        # Führe zusammen, wenn eine der Distanzen unter dem Schwellwert liegt
        return (
            horizontal_distance <= self.merge_distance
            or vertical_distance <= self.merge_distance
        )

    def crop_sensitive_regions(
        self, pdf_path: str, output_dir: str, page_numbers: Optional[List[int]] = None
    ) -> Dict[str, List[str]]:
        """
        Croppt sensitive Regionen aus einem PDF und speichert sie als separate Bilder.

        Args:
            pdf_path: Pfad zum PDF
            output_dir: Ausgabeverzeichnis für gecropte Bilder
            page_numbers: Spezifische Seitenzahlen zu verarbeiten (None = alle Seiten)

        Returns:
            Dictionary mit Seite -> Liste von Crop-Bild-Pfaden
        """
        try:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            # Konvertiere PDF zu Bildern
            logger.info(f"Konvertiere PDF zu Bildern: {pdf_path}")
            images = convert_pdf_to_images(pdf_path)

            if page_numbers is None:
                page_numbers = list(range(len(images)))

            results = {}
            pdf_name = Path(pdf_path).stem

            for page_num in page_numbers:
                if page_num >= len(images):
                    logger.warning(f"Seite {page_num} existiert nicht in PDF")
                    continue

                logger.info(f"Verarbeite Seite {page_num + 1}")
                image = images[page_num]

                # Führe OCR durch, um Word-Boxes zu erhalten
                full_text, word_boxes = tesseract_full_image_ocr(image)

                if not word_boxes:
                    logger.warning(f"Keine Textboxen auf Seite {page_num + 1} gefunden")
                    continue

                # Erkenne sensitive Regionen
                sensitive_regions = self.detect_sensitive_regions(image, word_boxes)

                if not sensitive_regions:
                    logger.info(
                        f"Keine sensitiven Regionen auf Seite {page_num + 1} gefunden"
                    )
                    results[f"page_{page_num + 1}"] = []
                    continue

                # Croppe und speichere sensitive Regionen
                page_crops = []
                for i, (x1, y1, x2, y2) in enumerate(sensitive_regions):
                    # Croppe die Region
                    cropped_image = image.crop((x1, y1, x2, y2))

                    # Erstelle Dateinamen
                    crop_filename = f"{pdf_name}_page_{page_num + 1}_region_{i + 1}.png"
                    crop_path = str(output_dir_path / Path(crop_filename))

                    # Speichere das gecropte Bild
                    cropped_image.save(crop_path)
                    page_crops.append(str(crop_path))

                    logger.info(
                        f"Gespeichert: {crop_filename} ({x2 - x1}x{y2 - y1} px)"
                    )

                results[f"page_{page_num + 1}"] = page_crops
        except Exception as e:
            logger.error(f"Fehler beim Cropping von Seite {page_num + 1}: {e}")
            results[f"page_{page_num + 1}"] = []

        return results

    def visualize_sensitive_regions(
        self,
        image: Image.Image,
        word_boxes: List[Tuple[str, Tuple[int, int, int, int]]],
        output_path: str,
    ) -> None:
        """
        Visualisiert erkannte sensitive Regionen auf dem Bild für Debugging.

        Args:
            image: PIL Image
            word_boxes: OCR Word-Boxes
            output_path: Pfad für das Ausgabebild
        """
        # Kopiere das Bild für Visualisierung
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)

        # Erkenne sensitive Regionen
        sensitive_regions = self.detect_sensitive_regions(image, word_boxes)

        # Zeichne sensitive Regionen in Rot
        for i, (x1, y1, x2, y2) in enumerate(sensitive_regions):
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 20), f"Region {i + 1}", fill="red")

        # Zeichne alle OCR Word-Boxes in Blau (optional)
        for word, (x, y, w, h) in word_boxes:
            draw.rectangle([x, y, x + w, y + h], outline="blue", width=1)

        vis_image.save(output_path)
        logger.info(f"Visualisierung gespeichert: {output_path}")

    def create_anonymized_pdf_with_crops(
        self, pdf_path: str, crop_output_dir: str, anonymized_pdf_path: str
    ) -> None:
        """
        Erstellt eine anonymisierte Version des PDFs, bei der sensitive Regionen
        durch schwarze Rechtecke überdeckt werden.

        Args:
            pdf_path: Ursprüngliches PDF
            crop_output_dir: Verzeichnis mit den Crop-Daten
            anonymized_pdf_path: Pfad für das anonymisierte PDF
        """
        try:
            import fitz  # PyMuPDF

            logger.info(f"Erstelle anonymisiertes PDF: {anonymized_pdf_path}")

            # Öffne das ursprüngliche PDF
            doc = fitz.open(pdf_path)

            # Konvertiere PDF zu Bildern für die Analyse
            images = convert_pdf_to_images(pdf_path)

            # Verarbeite jede Seite
            for page_num, image in enumerate(images):
                page = doc[page_num]

                # Führe OCR durch, um sensitive Bereiche zu finden
                full_text, word_boxes = tesseract_full_image_ocr(image)

                # Erkenne sensitive Regionen
                sensitive_regions = self.detect_sensitive_regions(image, word_boxes)

                if sensitive_regions:
                    logger.info(
                        f"Anonymisiere {len(sensitive_regions)} Bereiche auf Seite {page_num + 1}"
                    )

                    # Konvertiere Pixel-Koordinaten zu PDF-Koordinaten
                    # PDF-Koordinaten haben Ursprung unten links, Bilder oben links
                    page_rect = page.rect
                    page_height = page_rect.height
                    page_width = page_rect.width

                    img_width, img_height = image.size

                    # Skalierungsfaktoren
                    scale_x = page_width / img_width
                    scale_y = page_height / img_height

                    for x1, y1, x2, y2 in sensitive_regions:
                        # Konvertiere Bildkoordinaten zu PDF-Koordinaten
                        pdf_x1 = x1 * scale_x
                        pdf_y1 = page_height - (y2 * scale_y)  # Y-Achse umkehren
                        pdf_x2 = x2 * scale_x
                        pdf_y2 = page_height - (y1 * scale_y)  # Y-Achse umkehren

                        # Erstelle schwarzes Rechteck
                        rect = fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)

                        # Füge schwarzes Rechteck hinzu
                        page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))

                        logger.debug(
                            f"Geschwärzt: ({pdf_x1:.1f}, {pdf_y1:.1f}, {pdf_x2:.1f}, {pdf_y2:.1f})"
                        )

            # Speichere das anonymisierte PDF
            doc.save(anonymized_pdf_path)
            doc.close()

            logger.info(f"Anonymisiertes PDF gespeichert: {anonymized_pdf_path}")

        except ImportError:
            logger.error("PymuPDF not installed.")
            raise
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des anonymisierten PDFs: {e}")
            raise


def crop_sensitive_regions_from_pdf(
    pdf_path: str, output_dir: str, margin: int = 20, visualize: bool = False
) -> Dict[str, List[str]]:
    """
    Convenience-Funktion zum Cropping sensitiver Regionen aus einem PDF.

    Args:
        pdf_path: Pfad zum PDF
        output_dir: Ausgabeverzeichnis
        margin: Zusätzlicher Rand um sensitive Bereiche
        visualize: Ob Visualisierungen erstellt werden sollen

    Returns:
        Dictionary mit gecropten Bildpfaden pro Seite
    """
    cropper = SensitiveRegionCropper(margin=margin)

    results = cropper.crop_sensitive_regions(pdf_path, output_dir)

    if visualize:
        # Erstelle Visualisierungen für Debugging
        vis_dir = Path(output_dir) / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        images = convert_pdf_to_images(pdf_path)
        pdf_name = Path(pdf_path).stem

        for page_num, image in enumerate(images):
            full_text, word_boxes = tesseract_full_image_ocr(image)
            vis_path = vis_dir / f"{pdf_name}_page_{page_num + 1}_visualization.png"
            cropper.visualize_sensitive_regions(image, word_boxes, str(vis_path))

    return results
