#!/usr/bin/env python3
"""
Test script to verify OCR improvements with enhanced preprocessing and gibberish filtering.

This script tests:
1. Enhanced preprocessing (denoising, CLAHE, sharpening, adaptive thresholding)
2. Gibberish detection and filtering
3. Higher confidence thresholds
4. Better PSM selection
"""

import cv2
import numpy as np

from lx_anonymizer.setup.custom_logger import get_logger
from lx_anonymizer.ocr.ocr_frame_tesserocr import TesseOCRFrameProcessor

# Setup logging
logger = get_logger(__name__)


def create_test_image_with_text(
    text: str, size=(400, 100), noise_level=0
) -> np.ndarray:
    """Create a test image with text and optional noise."""
    img = np.ones(size, dtype=np.uint8) * 255  # White background

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 60), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, size).astype(np.uint8)
        img = cv2.add(img, noise)

    return img


def test_clean_text():
    """Test OCR on clean text."""
    print("\n" + "=" * 60)
    print("TEST 1: Clean Text")
    print("=" * 60)

    processor = TesseOCRFrameProcessor()
    img = create_test_image_with_text("2024-01-15 14:30:25")

    roi = {"x": 0, "y": 0, "width": img.shape[1], "height": img.shape[0]}
    text, conf, _ = processor.extract_text_from_frame(img, roi=roi, high_quality=True)

    print("Input text: '2024-01-15 14:30:25'")
    print(f"OCR result: '{text}'")
    print(f"Confidence: {conf:.2f}%")
    print(f"Status: {'✓ PASS' if '2024' in text else '✗ FAIL'}")

    return "2024" in text


def test_noisy_text():
    """Test OCR on noisy text with preprocessing."""
    print("\n" + "=" * 60)
    print("TEST 2: Noisy Text (with enhanced preprocessing)")
    print("=" * 60)

    processor = TesseOCRFrameProcessor()
    img = create_test_image_with_text("Patient: John Doe", noise_level=30)

    roi = {"x": 0, "y": 0, "width": img.shape[1], "height": img.shape[0]}
    text, conf, _ = processor.extract_text_from_frame(img, roi=roi, high_quality=True)

    print("Input text: 'Patient: John Doe' (with noise)")
    print(f"OCR result: '{text}'")
    print(f"Confidence: {conf:.2f}%")
    print(
        f"Status: {'✓ PASS' if text and len(text) > 5 else '✗ FAIL (filtered as gibberish)'}"
    )

    return text and len(text) > 5


def test_gibberish_filtering():
    """Test that gibberish is properly filtered."""
    print("\n" + "=" * 60)
    print("TEST 3: Gibberish Filtering")
    print("=" * 60)

    processor = TesseOCRFrameProcessor()

    # Create very noisy image that produces gibberish
    img = np.random.randint(0, 256, (100, 400), dtype=np.uint8)

    roi = {"x": 0, "y": 0, "width": img.shape[1], "height": img.shape[0]}
    text, conf, _ = processor.extract_text_from_frame(img, roi=roi, high_quality=True)

    print("Input: Random noise image")
    print(f"OCR result: '{text}'")
    print(f"Confidence: {conf:.2f}%")
    print(
        f"Status: {'✓ PASS (gibberish filtered)' if not text or len(text) < 5 else '✗ FAIL (should filter gibberish)'}"
    )

    return not text or len(text) < 5


def test_roi_processing():
    """Test ROI-based processing with multiple regions."""
    print("\n" + "=" * 60)
    print("TEST 4: Multiple ROI Processing")
    print("=" * 60)

    processor = TesseOCRFrameProcessor()

    # Create image with two text regions
    img = np.ones((200, 600), dtype=np.uint8) * 255
    cv2.putText(img, "2024-01-15", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "ID: 12345", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    roi1 = {"x": 0, "y": 0, "width": 300, "height": 100}
    roi2 = {"x": 0, "y": 100, "width": 300, "height": 100}

    text1, conf1, _ = processor.extract_text_from_frame(
        img, roi=roi1, high_quality=True
    )
    text2, conf2, _ = processor.extract_text_from_frame(
        img, roi=roi2, high_quality=True
    )

    print("ROI 1 - Expected: '2024-01-15'")
    print(f"ROI 1 - Result: '{text1}' (conf: {conf1:.2f}%)")
    print("ROI 2 - Expected: 'ID: 12345'")
    print(f"ROI 2 - Result: '{text2}' (conf: {conf2:.2f}%)")

    success = "2024" in text1 and ("12345" in text2 or "ID" in text2)
    print(f"Status: {'✓ PASS' if success else '✗ FAIL'}")

    return success


def test_low_confidence_filtering():
    """Test that low confidence results are filtered."""
    print("\n" + "=" * 60)
    print("TEST 5: Low Confidence Filtering")
    print("=" * 60)

    processor = TesseOCRFrameProcessor()

    # Create very low contrast image
    img = np.ones((100, 400), dtype=np.uint8) * 200
    cv2.putText(
        img, "Barely Visible", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 210, 210), 1
    )

    roi = {"x": 0, "y": 0, "width": img.shape[1], "height": img.shape[0]}
    text, conf, _ = processor.extract_text_from_frame(img, roi=roi, high_quality=True)

    print("Input: Very low contrast text")
    print(f"OCR result: '{text}'")
    print(f"Confidence: {conf:.2f}%")
    print("Confidence threshold: 40% (min) / 50% (high quality)")
    print(
        f"Status: {'✓ PASS (low confidence filtered)' if conf < 40 or not text else '✗ FAIL'}"
    )

    return conf < 40 or not text


def main():
    """Run all OCR improvement tests."""
    print("\n" + "=" * 70)
    print(" OCR IMPROVEMENTS TEST SUITE")
    print(" Testing: Enhanced Preprocessing + Gibberish Filtering")
    print("=" * 70)

    results = []

    try:
        results.append(("Clean Text", test_clean_text()))
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        results.append(("Clean Text", False))

    try:
        results.append(("Noisy Text", test_noisy_text()))
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        results.append(("Noisy Text", False))

    try:
        results.append(("Gibberish Filtering", test_gibberish_filtering()))
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        results.append(("Gibberish Filtering", False))

    try:
        results.append(("Multiple ROI", test_roi_processing()))
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        results.append(("Multiple ROI", False))

    try:
        results.append(("Low Confidence", test_low_confidence_filtering()))
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        results.append(("Low Confidence", False))

    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:25s}: {status}")

    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")
    print("=" * 70)

    if passed == total:
        print("\n🎉 All tests passed! OCR improvements are working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review results above.")

    return passed == total


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
