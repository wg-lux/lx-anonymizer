from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pytesseract  # type: ignore[import-untyped]
from PIL import Image

from lx_anonymizer.region_processing.box_operations import Box, OcrResult
from lx_anonymizer.region_processing.region_detector import (
    expand_roi,
)  # Ensure this module is correctly referenced
from lx_anonymizer.setup.custom_logger import get_logger

ImageInput: TypeAlias = str | Path | Image.Image
ArrayImageInput: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.integer[Any]]
TrocrImageInput: TypeAlias = ImageInput | ArrayImageInput


class _TesseractData(TypedDict):
    text: list[str]
    left: list[int | str]
    top: list[int | str]
    width: list[int | str]
    height: list[int | str]
    conf: list[int | str]


class _TensorLike(Protocol):
    def to(self, device: object) -> object: ...


class _ProcessorOutput(Protocol):
    pixel_values: _TensorLike


class _GenerateOutput(Protocol):
    sequences: object
    scores: list[object]


class _TrocrProcessorRuntime(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> _ProcessorOutput: ...

    def batch_decode(
        self, sequences: object, *, skip_special_tokens: bool = False
    ) -> list[str]: ...


class _TrocrTokenizerRuntime(Protocol):
    def batch_decode(
        self, sequences: object, *, skip_special_tokens: bool = False
    ) -> list[str]: ...


class _TrocrModelRuntime(Protocol):
    def cuda(self) -> "_TrocrModelRuntime": ...

    def to(self, device: object) -> "_TrocrModelRuntime": ...

    def generate(self, pixel_values: object, **kwargs: object) -> _GenerateOutput: ...


# Import CRAFT text detection if available (requires hezar)
try:
    from lx_anonymizer.text_detection.craft_text_detection import craft_text_detection

    _craft_available = True
except ImportError:
    _craft_available = False

    def craft_text_detection(
        image_input: Any,
        min_confidence: Any = None,
        width: Any = None,
        height: Any = None,
    ) -> Any:
        raise ImportError(
            "CRAFT text detection requires 'hezar' package. Install with: pip install lx-anonymizer[llm]"
        )


CRAFT_AVAILABLE: bool = _craft_available


logger = get_logger(__name__)
torch: Any

try:
    import torch as torch_mod  # type: ignore[import-untyped]

    torch = torch_mod
    _torch_available = True
except ImportError:
    torch = None
    _torch_available = False

TORCH_AVAILABLE: bool = _torch_available

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore[import-untyped]

    _transformers_available = True
except ImportError:
    TrOCRProcessor = VisionEncoderDecoderModel = None  # type: ignore[assignment]
    _transformers_available = False

TRANSFORMERS_AVAILABLE: bool = _transformers_available

try:
    import tesserocr  # type: ignore[import-untyped]  # noqa: F401

    _tesserocr_available = True
except ImportError:
    _tesserocr_available = False

TESSEROCR_AVAILABLE: bool = _tesserocr_available


def _load_rgb_image(image_input: ImageInput) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    return Image.open(image_input).convert("RGB")


def _load_rgb_trocr_image(image_input: TrocrImageInput) -> Image.Image:
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input).convert("RGB")
    return _load_rgb_image(image_input)


def _pytesseract_image_to_string(image: Image.Image, config: str = "") -> str:
    raw_output = cast(
        object,
        pytesseract.image_to_string(image, config=config),  # pyright: ignore[reportUnknownMemberType]
    )
    if isinstance(raw_output, bytes):
        return raw_output.decode(errors="replace")
    if isinstance(raw_output, str):
        return raw_output
    return str(raw_output)


def _pytesseract_image_to_data(image: Image.Image) -> _TesseractData:
    return cast(
        _TesseractData,
        pytesseract.image_to_data(  # pyright: ignore[reportUnknownMemberType]
            image, output_type=pytesseract.Output.DICT
        ),
    )


def _as_trocr_processor(processor: object) -> _TrocrProcessorRuntime:
    return cast(_TrocrProcessorRuntime, processor)


def _as_trocr_model(model: object) -> _TrocrModelRuntime:
    return cast(_TrocrModelRuntime, model)


def _as_trocr_tokenizer(tokenizer: object) -> _TrocrTokenizerRuntime:
    return cast(_TrocrTokenizerRuntime, tokenizer)


def _trocr_dependencies_available() -> bool:
    return TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE


def _get_model_service() -> object | None:
    if not _trocr_dependencies_available():
        return None
    try:
        from lx_anonymizer.model_service import model_service  # type: ignore[import-untyped]

        return model_service
    except ImportError:
        return None


def preload_models() -> tuple[_TrocrProcessorRuntime, _TrocrModelRuntime, object]:
    global processor, model, device

    if not _trocr_dependencies_available():
        raise ImportError(
            "TrOCR dependencies are not installed. Install with: pip install lx-anonymizer[ocr]"
        )

    logger.info("Preloading models...")

    # More explicit CUDA availability check
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Enable CUDA optimization
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    # Load models with CUDA memory optimization
    processor_cls = cast(Any, TrOCRProcessor)
    model_cls = cast(Any, VisionEncoderDecoderModel)
    processor = _as_trocr_processor(
        processor_cls.from_pretrained("microsoft/trocr-base-str")
    )
    model = _as_trocr_model(model_cls.from_pretrained("microsoft/trocr-base-str"))

    # Move model to GPU and enable CUDA optimizations
    if torch.cuda.is_available():
        model = model.cuda()

    model.to(device)

    logger.info("Models preloaded successfully.")
    return processor, model, device


def cleanup_gpu() -> None:
    """Clean up GPU memory"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_info() -> bool:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        logger.info(f"GPU Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        cudasupport = True
        return cudasupport
    else:
        cudasupport = False
        return cudasupport


def tesseract_full_image_ocr(
    image_path: ImageInput,
) -> tuple[str, list[OcrResult]]:
    """
    Perform OCR on the entire image using Tesseract.
    Returns:
      - A single string with all recognized text.
      - A list of (word, (left, top, width, height)) for each recognized word.
    """
    image = _load_rgb_image(image_path)
    full_text = _pytesseract_image_to_string(image, config="--psm 6")

    # 2. Word-level bounding boxes
    data = _pytesseract_image_to_data(image)
    word_boxes: list[OcrResult] = []
    for i in range(len(data["text"])):
        word = str(data["text"][i]).strip()
        if word != "":
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            word_boxes.append((word, (x, y, x + w, y + h)))

    return full_text.strip(), word_boxes


def trocr_full_image_ocr(image_input: TrocrImageInput) -> str:
    """
    Perform OCR on the entire image using TrOCR.
    Accepts:
      - numpy.ndarray (BGR/RGB/Gray)
      - PIL.Image.Image
      - Pfad/Datei-ähnliches Objekt
    Fallback zu Tesseract, wenn TrOCR-Modelle nicht verfügbar sind.
    """
    # 1) In PIL.Image umwandeln
    try:
        pil_img = _load_rgb_trocr_image(image_input)
    except Exception as e:
        logger.warning(
            f"Failed to prepare image for TrOCR: {e}. Falling back to Tesseract."
        )
        try:
            fb_img = _load_rgb_trocr_image(image_input)
            return _pytesseract_image_to_string(fb_img, config="--psm 6").strip()
        except Exception:
            return ""

    # 2) Modelle laden
    service = _get_model_service()
    if service is None:
        logger.info("TrOCR dependencies unavailable, falling back to Tesseract")
        return _pytesseract_image_to_string(pil_img, config="--psm 6").strip()

    load_trocr_model = cast(Any, service).load_trocr_model
    processor_raw, model_raw, _tokenizer, device = load_trocr_model()
    if processor_raw is None or model_raw is None:
        logger.warning("TrOCR model not available, falling back to Tesseract")
        return _pytesseract_image_to_string(pil_img, config="--psm 6").strip()
    processor = _as_trocr_processor(processor_raw)
    model = _as_trocr_model(model_raw)

    # 3) Inferenz
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(
        device
    )
    outputs = model.generate(
        pixel_values,
        output_scores=True,
        do_sample=False,
        num_beams=5,
        return_dict_in_generate=True,
        max_new_tokens=512,
    )
    return processor.batch_decode(outputs.sequences, skip_special_tokens=True)[
        0
    ].strip()


def trocr_full_image_ocr_on_boxes(image_path: ImageInput) -> str:
    """
    Perform OCR on the entire image using TrOCR.
    """
    # Lade Modelle vom Service anstatt von preload_models
    service = _get_model_service()
    if service is None:
        logger.info("TrOCR dependencies unavailable, falling back to tesseract")
        image = _load_rgb_image(image_path)
        full_text = _pytesseract_image_to_string(image, config="--psm 6")
        return full_text.strip()

    load_trocr_model = cast(Any, service).load_trocr_model
    processor_raw, model_raw, tokenizer_raw, device = load_trocr_model()

    # Behandle Fehler, wenn Modelle nicht geladen werden konnten
    if processor_raw is None or model_raw is None or tokenizer_raw is None:
        logger.warning("TrOCR model not available, falling back to tesseract")
        # Fallback zu Tesseract implementieren
        image = _load_rgb_image(image_path)
        full_text = _pytesseract_image_to_string(image, config="--psm 6")
        return full_text.strip()
    processor = _as_trocr_processor(processor_raw)
    model = _as_trocr_model(model_raw)
    tokenizer = _as_trocr_tokenizer(tokenizer_raw)

    image = _load_rgb_image(image_path)

    # Detect text regions using CRAFT
    boxes, _ = cast(tuple[list[Box], object], craft_text_detection(image_path))

    ocr_results: list[str] = []

    if boxes:
        logger.info(
            f"CRAFT detected {len(boxes)} regions. Processing each region with TrOCR."
        )
        for box in boxes:
            (startX, startY, endX, endY) = box
            cropped_image = image.crop((startX, startY, endX, endY))
            pixel_values = processor(
                cropped_image, return_tensors="pt"
            ).pixel_values.to(device)
            outputs = model.generate(
                pixel_values,
                output_scores=True,
                do_sample=True,
                temperature=0.6,
                return_dict_in_generate=True,
                max_new_tokens=50,
            )
            recognized_text = tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )[0]
            logger.debug(f"Box {box} yielded text: '{recognized_text}'")
            if recognized_text.strip():
                ocr_results.append(recognized_text.strip())
        final_text = "\n".join(ocr_results)
        if not final_text.strip():
            logger.warning(
                "No text recognized in regions, falling back to full image OCR."
            )
            final_text = trocr_full_image_ocr(image)
    else:
        logger.info("No regions detected by CRAFT. Falling back to full image OCR.")
        final_text = trocr_full_image_ocr(image)

    return final_text


def trocr_on_boxes(
    image_path: ImageInput,
    boxes: list[Box],
) -> tuple[list[OcrResult], list[float]]:
    try:
        if not _trocr_dependencies_available():
            logger.info("TrOCR dependencies unavailable, falling back to pytesseract")
            return tesseract_on_boxes_pytesseract(image_path, boxes)

        image = _load_rgb_image(image_path)
        extracted_text_with_boxes: list[OcrResult] = []
        confidences: list[float] = []
        # Ensure models are loaded
        processor, model, device = preload_models()

        logger.debug("Processing image with TrOCR")

        for idx, box in enumerate(boxes):
            cudasupport = print_gpu_info()
            try:
                (startX, startY, endX, endY) = box

                # Expand the region of interest
                image_np = np.asarray(image)
                image_shape = image_np.shape
                expanded_box = expand_roi(startX, startY, endX, endY, 5, image_shape)
                (startX_exp, startY_exp, endX_exp, endY_exp) = expanded_box

                # Crop the image to the expanded box
                cropped_image = image.crop((startX_exp, startY_exp, endX_exp, endY_exp))

                # Process the cropped image using the processor
                if cudasupport and TORCH_AVAILABLE:
                    # Use CUDA with automatic mixed precision
                    with torch.amp.autocast(device_type="cuda"):
                        pixel_values = processor(
                            images=cropped_image, return_tensors="pt"
                        ).pixel_values.to(device)

                        # Generate text with CUDA optimizations
                        outputs = model.generate(
                            pixel_values,
                            output_scores=True,
                            return_dict_in_generate=True,
                            max_new_tokens=50,
                        )
                else:
                    # CPU fallback
                    pixel_values = processor(
                        images=cropped_image, return_tensors="pt"
                    ).pixel_values

                    # Generate text without CUDA optimizations
                    outputs = model.generate(
                        pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_new_tokens=50,
                    )

                # Decode the output tokens into readable text
                generated_text = processor.batch_decode(
                    outputs.sequences, skip_special_tokens=True
                )[0]

                # Calculate confidence score from the last token's scores
                scores = outputs.scores  # List of logits for each generation step
                if scores:
                    # Take the scores from the last generation step
                    last_scores = scores[-1]
                    confidence_score = cast(
                        float,
                        torch.nn.functional.softmax(last_scores, dim=-1).max().item(),
                    )
                else:
                    confidence_score = (
                        0.0  # Default confidence if scores are unavailable
                    )

                # Append results to the lists
                extracted_text_with_boxes.append((generated_text.strip(), expanded_box))
                confidences.append(confidence_score)

                logger.info(
                    f"Processed box {idx + 1}/{len(boxes)}: '{generated_text.strip()}' with confidence {confidence_score:.4f}"
                )

            except Exception as e:
                logger.info(f"Error processing box {idx + 1}/{len(boxes)}: {e}")
                extracted_text_with_boxes.append(("", box))
                confidences.append(0.0)

        logger.debug("TrOCR processing complete")
        return extracted_text_with_boxes, confidences
    except Exception as e:
        logger.error(f"Error in TrOCR processing: {e}")
        cleanup_gpu()
        return [], []  # Added return for this exception path
    finally:
        cleanup_gpu()


def fallback_full_ocr(
    image: ImageInput,
    processor: _TrocrProcessorRuntime,
    model: _TrocrModelRuntime,
    device: object,
) -> str:
    """
    Fallback OCR on the entire image when region detection fails
    """
    image = _load_rgb_image(image)
    try:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
            device
        )
        outputs = model.generate(pixel_values, max_new_tokens=150)
        return processor.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception as e:
        logger.error(f"Fallback OCR failed: {e}")
        return ""


def tesseract_on_boxes(
    image_path: ImageInput,
    boxes: list[Box],
    use_fast_ocr: bool = True,
) -> tuple[list[OcrResult], list[float]]:
    """
    Enhanced tesseract_on_boxes with automatic optimization.

    Args:
        image_path: Path to image or PIL Image object
        boxes: List of bounding boxes (startX, startY, endX, endY)
        use_fast_ocr: If True, use TesseOCR when available for 10-50x speedup

    Returns:
        Tuple of (extracted_text_with_boxes, confidences)
    """
    # Use optimized TesseOCR if available and requested
    if use_fast_ocr and TESSEROCR_AVAILABLE:
        try:
            logger.debug("Using optimized TesseOCR for text extraction")
            return tesseract_on_boxes_pytesseract(image_path, boxes)
        except Exception as e:
            logger.warning(f"TesseOCR failed ({e}), falling back to pytesseract")

    # Fallback to original pytesseract implementation
    logger.debug("Using pytesseract for text extraction")
    return tesseract_on_boxes_pytesseract(image_path, boxes)


def tesseract_on_boxes_pytesseract(
    image_path: ImageInput,
    boxes: list[Box],
) -> tuple[list[OcrResult], list[float]]:
    """
    Original pytesseract implementation (renamed for clarity).

    This is the original function that uses subprocess calls to tesseract CLI.
    Kept for compatibility and as a fallback when TesseOCR is not available.
    """
    image = _load_rgb_image(image_path)

    extracted_text_with_boxes: list[OcrResult] = []
    confidences: list[float] = []

    logger.debug("Processing image with Tesseract OCR")

    for idx, box in enumerate(boxes):
        try:
            cropped_image = image.crop(box)

            # Use pytesseract to perform OCR on the cropped image
            ocr_result = _pytesseract_image_to_string(cropped_image, config="--psm 6")

            # Get confidence scores from pytesseract
            details = _pytesseract_image_to_data(cropped_image)
            text_confidences = [
                int(conf) for conf in details["conf"] if str(conf).isdigit()
            ]

            # Calculate the average confidence score
            confidence_score = (
                sum(text_confidences) / len(text_confidences)
                if text_confidences
                else 0.0
            )

            # Append results to the lists
            extracted_text_with_boxes.append((ocr_result.strip(), box))
            confidences.append(confidence_score)

            logger.debug(
                f"Processed box {idx + 1}/{len(boxes)}: '{ocr_result.strip()}' with confidence {confidence_score:.2f}"
            )

        except Exception as e:
            logger.info(f"Error processing box {idx + 1}/{len(boxes)}: {e}")
            extracted_text_with_boxes.append(("", box))
            confidences.append(0.0)

    logger.info("Tesseract OCR processing complete")
    return extracted_text_with_boxes, confidences


if __name__ == "__main__":
    # Preload models once when the script runs
    processor, model, device = preload_models()

    # Example usage:
    # Define your image path and bounding boxes
    image_path = "path/to/your/image.jpg"
    boxes = [
        (50, 50, 200, 150),
        (250, 80, 400, 180),
        # Add more boxes as needed
    ]

    # Perform OCR using TrOCR
    trocr_results, trocr_confidences = trocr_on_boxes(image_path, boxes)
    logger.debug("TrOCR Results:", trocr_results)
    logger.debug("TrOCR Confidences:", trocr_confidences)

    # Perform OCR using Tesseract
    tesseract_results, tesseract_confidences = tesseract_on_boxes(image_path, boxes)
    logger.debug("Tesseract Results:", tesseract_results)
    logger.debug("Tesseract Confidences:", tesseract_confidences)
    logger.debug("Tesseract Confidences:", tesseract_confidences)
    logger.debug("Tesseract Confidences:", tesseract_confidences)
    logger.debug("Tesseract Confidences:", tesseract_confidences)
