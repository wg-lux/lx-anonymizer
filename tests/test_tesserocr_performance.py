#!/usr/bin/env python3
"""
Performance test script for TesseOCR vs pytesseract in video OCR processing.

This script demonstrates the significant performance improvement when using
TesseOCR instead of pytesseract for video frame text extraction.
"""

import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

# Add the libs path to sys.path
sys.path.insert(0, "/home/admin/dev/lx-annotate/libs/lx-anonymizer")

try:
    from lx_anonymizer.ocr_frame import FrameOCR
    from lx_anonymizer.ocr_frame_tesserocr import TesseOCRFrameProcessor

    print("✅ Successfully imported OCR modules")
except ImportError as e:
    print(f"❌ Failed to import OCR modules: {e}")
    sys.exit(1)


def create_test_frame_with_text():
    """Create a synthetic video frame with medical text overlay."""
    # Create a frame similar to endoscopy video
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 100, 80)  # Dark greenish background

    # Add some medical text overlay (simulated)
    cv2.putText(
        frame,
        "Patient: Max Mustermann",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "Geburtsdatum: 15.03.1975",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "Fallnr: 12345",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "Datum: 13.08.2025",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )

    return frame


def benchmark_ocr_performance():
    """Benchmark OCR performance comparison."""
    print("🚀 Starting OCR Performance Benchmark")
    print("=" * 50)

    # Create test frame
    test_frame = create_test_frame_with_text()

    # Define test parameters
    num_frames = 10  # Process 10 frames to measure average performance
    roi = {"x": 10, "y": 10, "width": 400, "height": 150}  # Focus on text area

    print(f"Test setup: {num_frames} frames, ROI: {roi}")
    print()

    # Test pytesseract-based FrameOCR
    print("🐌 Testing pytesseract-based FrameOCR...")
    frame_ocr_pytesseract = FrameOCR(use_fast_ocr=False)

    start_time = time.time()
    pytesseract_results = []

    for i in range(num_frames):
        text, confidence, metadata = frame_ocr_pytesseract.extract_text_from_frame(
            test_frame, roi, high_quality=True
        )
        pytesseract_results.append((text, confidence))
        print(f"  Frame {i + 1}: '{text[:30]}...' (conf: {confidence:.2f})")

    pytesseract_time = time.time() - start_time
    print(f"✅ pytesseract total time: {pytesseract_time:.3f}s")
    print(f"   Average per frame: {pytesseract_time / num_frames:.3f}s")
    print()

    # Test TesseOCR-based FrameOCR
    print("🚀 Testing TesseOCR-based FrameOCR...")
    frame_ocr_tesserocr = FrameOCR(use_fast_ocr=True)

    start_time = time.time()
    tesserocr_results = []

    for i in range(num_frames):
        text, confidence, metadata = frame_ocr_tesserocr.extract_text_from_frame(
            test_frame, roi, high_quality=True
        )
        tesserocr_results.append((text, confidence))
        print(f"  Frame {i + 1}: '{text[:30]}...' (conf: {confidence:.2f})")

    tesserocr_time = time.time() - start_time
    print(f"✅ TesseOCR total time: {tesserocr_time:.3f}s")
    print(f"   Average per frame: {tesserocr_time / num_frames:.3f}s")
    print()

    # Calculate performance improvement
    if tesserocr_time > 0:
        speedup = pytesseract_time / tesserocr_time
        print("📊 PERFORMANCE RESULTS")
        print("=" * 30)
        print(
            f"🐌 pytesseract: {pytesseract_time:.3f}s ({pytesseract_time / num_frames:.3f}s/frame)"
        )
        print(
            f"🚀 TesseOCR:    {tesserocr_time:.3f}s ({tesserocr_time / num_frames:.3f}s/frame)"
        )
        print(f"⚡ Speedup:     {speedup:.1f}x faster!")
        print()

        # Estimate performance for video processing
        frames_per_second_pytesseract = 1.0 / (pytesseract_time / num_frames)
        frames_per_second_tesserocr = 1.0 / (tesserocr_time / num_frames)

        print("📹 VIDEO PROCESSING ESTIMATES")
        print("=" * 35)
        print(f"📺 pytesseract: {frames_per_second_pytesseract:.1f} frames/second")
        print(f"⚡ TesseOCR:    {frames_per_second_tesserocr:.1f} frames/second")
        print()
        print(f"For a 60-second video at 30 FPS (1800 frames):")
        print(
            f"🐌 pytesseract would take: {1800 * (pytesseract_time / num_frames):.1f} seconds"
        )
        print(
            f"🚀 TesseOCR would take:    {1800 * (tesserocr_time / num_frames):.1f} seconds"
        )
        print(
            f"⏰ Time saved:            {1800 * (pytesseract_time / num_frames) - 1800 * (tesserocr_time / num_frames):.1f} seconds"
        )

    # Test direct TesseOCR processor
    print("\n🔬 Testing direct TesseOCR processor...")
    direct_processor = TesseOCRFrameProcessor()

    start_time = time.time()
    for i in range(num_frames):
        text, confidence, metadata = direct_processor.extract_text_from_frame(
            test_frame, roi, high_quality=True
        )
    direct_time = time.time() - start_time

    print(
        f"✅ Direct TesseOCR time: {direct_time:.3f}s ({direct_time / num_frames:.3f}s/frame)"
    )

    # Performance stats
    stats = direct_processor.get_performance_stats()
    print(f"📈 Performance stats: {stats}")


if __name__ == "__main__":
    try:
        benchmark_ocr_performance()
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        traceback.print_exc()
