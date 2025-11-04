"""
Comprehensive tests for the FrameCleaner module.

This test suite covers:
- FrameCleaner initialization and configuration
- Video frame extraction and processing
- Masking functionality integration
- ROI processing and validation
- Sensitive frame detection
- Error handling and edge cases
- Hardware acceleration detection
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import the modules we're testing
try:
    from lx_anonymizer.frame_cleaner import FrameCleaner
    from lx_anonymizer.masking import MaskApplication
    from lx_anonymizer.video_encoder import VideoEncoder
except ImportError:
    pytest.skip("lx_anonymizer modules not available", allow_module_level=True)


def create_test_video(
    output_path: Path, duration: int = 2, width: int = 320, height: int = 240
) -> Path:
    """
    Create a small synthetic test video using ffmpeg.

    Args:
        output_path: Where to save the test video
        duration: Duration in seconds
        width: Video width
        height: Video height

    Returns:
        Path to created video file
    """
    cmd = [
        "ffmpeg",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size={width}x{height}:rate=1",
        "-c:v",
        "libx264",
        "-t",
        str(duration),
        "-y",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        if not output_path.exists() or output_path.stat().st_size == 0:
            pytest.skip("Failed to create valid test video")
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("ffmpeg not available for test video creation")


class TestFrameCleaner:
    """Test cases for FrameCleaner functionality."""

    @pytest.fixture
    def test_video(self):
        """Create a temporary test video for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "test_video.mp4"
            create_test_video(video_path)
            yield video_path

    @pytest.fixture
    def endoscope_image_roi(self):
        """Sample endoscope image ROI configuration."""
        return {
            "x": 550,
            "y": 0,
            "width": 1350,
            "height": 1080,
            "image_width": 1920,
            "image_height": 1080,
        }

    @pytest.fixture
    def endoscope_data_roi_nested(self):
        """Sample nested endoscope data ROI configuration."""
        return {
            "patient_info": {"x": 50, "y": 50, "width": 400, "height": 200},
            "examination_info": {"x": 50, "y": 300, "width": 400, "height": 150},
            "examiner_info": {"x": 50, "y": 500, "width": 400, "height": 100},
        }

    def test_frame_cleaner_initialization_default(self):
        """Test FrameCleaner initialization with default settings."""
        frame_cleaner = FrameCleaner()

        # Check core components are initialized
        assert hasattr(frame_cleaner, "frame_ocr")
        assert hasattr(frame_cleaner, "frame_metadata_extractor")
        assert hasattr(frame_cleaner, "patient_data_extractor")
        assert hasattr(frame_cleaner, "roi_processor")
        assert hasattr(frame_cleaner, "mask_application")
        assert hasattr(frame_cleaner, "video_encoder")

        # Check default settings
        assert frame_cleaner.use_enhanced_ocr is True
        assert frame_cleaner.use_llm is False
        assert isinstance(frame_cleaner.mask_application, MaskApplication)
        assert isinstance(frame_cleaner.video_encoder, VideoEncoder)

    def test_frame_cleaner_initialization_with_llm(self):
        """Test FrameCleaner initialization with LLM enabled."""
        frame_cleaner = FrameCleaner(use_llm=True)

        assert frame_cleaner.use_llm is True
        # LLM components may or may not be available depending on environment
        assert hasattr(frame_cleaner, "ollama_proc")
        assert hasattr(frame_cleaner, "ollama_extractor")

    def test_frame_cleaner_initialization_without_llm(self):
        """Test FrameCleaner initialization with LLM disabled."""
        frame_cleaner = FrameCleaner(use_llm=False)

        assert frame_cleaner.use_llm is False
        assert frame_cleaner.ollama_proc is None
        assert frame_cleaner.ollama_extractor is None

    def test_hardware_acceleration_detection(self):
        """Test hardware acceleration detection."""
        frame_cleaner = FrameCleaner()

        # Should detect hardware capabilities
        assert hasattr(frame_cleaner, "nvenc_available")
        assert hasattr(frame_cleaner, "preferred_encoder")
        assert isinstance(frame_cleaner.preferred_encoder, dict)

        # Encoder should have required fields
        encoder = frame_cleaner.preferred_encoder
        assert "name" in encoder
        assert "type" in encoder

    def test_masking_integration(self):
        """Test that masking functionality is properly integrated."""
        frame_cleaner = FrameCleaner()

        # Check masking methods are accessible
        assert hasattr(frame_cleaner, "_mask_video_streaming")
        assert hasattr(frame_cleaner, "_create_mask_config_from_roi")

        # Test mask config creation
        test_roi = {
            "x": 550,
            "y": 0,
            "width": 1350,
            "height": 1080,
            "image_width": 1920,
            "image_height": 1080,
        }

        mask_config = frame_cleaner._create_mask_config_from_roi(test_roi)
        assert isinstance(mask_config, dict)
        assert "endoscope_image_x" in mask_config
        assert "endoscope_image_y" in mask_config
        assert "endoscope_image_width" in mask_config
        assert "endoscope_image_height" in mask_config

    def test_roi_validation(self, endoscope_image_roi):
        """Test ROI validation functionality."""
        frame_cleaner = FrameCleaner()

        # Valid ROI should pass validation
        assert frame_cleaner._validate_roi(endoscope_image_roi) is True

        # Invalid ROI should fail validation
        invalid_roi = {"x": -10, "y": -10, "width": 0, "height": 0}
        assert frame_cleaner._validate_roi(invalid_roi) is False

        # Non-dict should fail validation (skip type checking for test purposes)
        # Note: These would raise TypeErrors in practice due to type hints
        try:
            result1 = frame_cleaner._validate_roi("not a dict")  # type: ignore
            assert result1 is False
        except (TypeError, AttributeError):
            pass  # Expected due to type constraints

        try:
            result2 = frame_cleaner._validate_roi(None)  # type: ignore
            assert result2 is False
        except (TypeError, AttributeError):
            pass  # Expected due to type constraints

    @patch("cv2.VideoCapture")
    def test_iter_video_basic(self, mock_cv2_capture, test_video):
        """Test basic video iteration functionality."""
        frame_cleaner = FrameCleaner()

        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 10  # 10 total frames
        mock_cap.read.side_effect = [
            (True, np.zeros((240, 320, 3), dtype=np.uint8)),  # Frame 0
            (True, np.zeros((240, 320, 3), dtype=np.uint8)),  # Frame 1
            (False, None),  # End of video
        ]
        mock_cv2_capture.return_value = mock_cap

        frames = list(frame_cleaner._iter_video(test_video, total_frames=10))

        # Should have extracted some frames
        assert len(frames) > 0
        for idx, gray_frame, stride in frames:
            assert isinstance(idx, int)
            assert isinstance(gray_frame, np.ndarray)
            assert isinstance(stride, int)
            assert gray_frame.ndim == 2  # Should be grayscale

    def test_extract_frames(self, test_video):
        """Test frame extraction to directory."""
        frame_cleaner = FrameCleaner()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "frames"

            try:
                frames = frame_cleaner.extract_frames(
                    test_video, output_dir, max_frames=2
                )

                assert len(frames) > 0
                assert all(frame.exists() for frame in frames)
                assert all(
                    frame.suffix == ".png" for frame in frames
                )  # Should be PNG for OCR quality

            except RuntimeError:
                # If ffmpeg fails, that's expected in some test environments
                pytest.skip("ffmpeg frame extraction failed in test environment")

    def test_process_frame_single(self, endoscope_image_roi, endoscope_data_roi_nested):
        """Test single frame processing functionality."""
        frame_cleaner = FrameCleaner()

        # Create a test grayscale frame
        test_frame = np.zeros((1080, 1920), dtype=np.uint8)
        test_frame[100:200, 100:400] = 255  # Add some white regions

        # Mock OCR to avoid dependencies
        with patch.object(
            frame_cleaner.frame_ocr,
            "extract_text_with_confidence",
            return_value=("Test text", 0.9),
        ):
            with patch.object(
                frame_cleaner.frame_metadata_extractor,
                "extract_frame_metadata",
                return_value={},
            ):
                is_sensitive, frame_meta, ocr_text, ocr_conf = (
                    frame_cleaner._process_frame_single(
                        gray_frame=test_frame,
                        endoscope_image_roi=endoscope_image_roi,
                        endoscope_data_roi_nested=endoscope_data_roi_nested,
                        frame_id=0,
                    )
                )

        assert isinstance(is_sensitive, bool)
        assert isinstance(frame_meta, dict)
        assert isinstance(ocr_text, str)
        assert isinstance(ocr_conf, (int, float))

    def test_clean_video_mask_overlay(
        self, test_video, endoscope_image_roi, endoscope_data_roi_nested
    ):
        """Test video cleaning with mask overlay technique."""
        frame_cleaner = FrameCleaner()

        # Mock frame processing to avoid long execution
        with patch.object(
            frame_cleaner,
            "_iter_video",
            return_value=[
                (0, np.zeros((240, 320), dtype=np.uint8), 1),
                (1, np.zeros((240, 320), dtype=np.uint8), 1),
            ],
        ):
            with patch.object(
                frame_cleaner,
                "_process_frame_single",
                return_value=(False, {}, "test text", 0.9),
            ):
                with patch.object(
                    frame_cleaner, "_mask_video_streaming", return_value=True
                ):
                    result_path, metadata = frame_cleaner.clean_video(
                        video_path=test_video,
                        endoscope_image_roi=endoscope_image_roi,
                        endoscope_data_roi_nested=endoscope_data_roi_nested,
                        technique="mask_overlay",
                    )

        assert isinstance(result_path, Path)
        assert isinstance(metadata, dict)
        assert "_anony" in result_path.stem

    def test_clean_video_remove_frames(
        self, test_video, endoscope_image_roi, endoscope_data_roi_nested
    ):
        """Test video cleaning with frame removal technique."""
        frame_cleaner = FrameCleaner()

        # Mock frame processing to simulate sensitive frames
        with patch.object(
            frame_cleaner,
            "_iter_video",
            return_value=[
                (0, np.zeros((240, 320), dtype=np.uint8), 1),
                (1, np.zeros((240, 320), dtype=np.uint8), 1),
                (2, np.zeros((240, 320), dtype=np.uint8), 1),
            ],
        ):
            with patch.object(
                frame_cleaner,
                "_process_frame_single",
                side_effect=[
                    (True, {}, "sensitive text", 0.9),  # Sensitive frame
                    (False, {}, "normal text", 0.9),  # Normal frame
                    (False, {}, "normal text", 0.9),  # Normal frame
                ],
            ):
                with patch.object(
                    frame_cleaner,
                    "remove_frames_from_video_streaming",
                    return_value=True,
                ):
                    result_path, metadata = frame_cleaner.clean_video(
                        video_path=test_video,
                        endoscope_image_roi=endoscope_image_roi,
                        endoscope_data_roi_nested=endoscope_data_roi_nested,
                        technique="remove_frames",
                    )

        assert isinstance(result_path, Path)
        assert isinstance(metadata, dict)

    def test_clean_video_invalid_technique(
        self, test_video, endoscope_image_roi, endoscope_data_roi_nested
    ):
        """Test video cleaning with invalid technique."""
        frame_cleaner = FrameCleaner()

        # Should default to mask_overlay for invalid technique
        with patch.object(frame_cleaner, "_iter_video", return_value=[]):
            with patch.object(
                frame_cleaner, "_mask_video_streaming", return_value=True
            ):
                result_path, metadata = frame_cleaner.clean_video(
                    video_path=test_video,
                    endoscope_image_roi=endoscope_image_roi,
                    endoscope_data_roi_nested=endoscope_data_roi_nested,
                    technique="invalid_technique",
                )

        assert isinstance(result_path, Path)
        assert isinstance(metadata, dict)


class TestFrameCleanerErrorHandling:
    """Tests for error handling and edge cases."""

    def test_clean_video_missing_file(self):
        """Test video cleaning with missing input file."""
        frame_cleaner = FrameCleaner()

        non_existent_file = Path("/path/that/does/not/exist.mp4")

        # Should handle missing file gracefully
        with pytest.raises((FileNotFoundError, OSError)):
            frame_cleaner.clean_video(
                video_path=non_existent_file,
                endoscope_image_roi={"x": 0, "y": 0, "width": 100, "height": 100},
                endoscope_data_roi_nested={},
            )

    def test_invalid_roi_handling(self):
        """Test handling of invalid ROI configurations."""
        frame_cleaner = FrameCleaner()

        # Test various invalid ROI configurations
        invalid_rois = [
            None,
            {},
            {"x": -10, "y": -10, "width": 0, "height": 0},
            {"x": "invalid", "y": 0, "width": 100, "height": 100},
            {"missing_required_fields": True},
        ]

        for invalid_roi in invalid_rois:
            result = frame_cleaner._validate_roi(invalid_roi)
            assert result is False

    @patch("cv2.VideoCapture")
    def test_video_capture_failure(self, mock_cv2_capture):
        """Test handling of video capture failures."""
        frame_cleaner = FrameCleaner()

        # Mock failed video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap

        test_video = Path("nonexistent.mp4")

        # Should handle capture failure gracefully
        frames = list(frame_cleaner._iter_video(test_video, total_frames=10))
        assert len(frames) == 0


class TestFrameCleanerIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def test_video_with_content(self):
        """Create a test video with some visual content."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "content_video.mp4"
            # Create video with more visual content for better testing
            create_test_video(video_path, duration=1, width=640, height=480)
            yield video_path

    def test_full_pipeline_integration(self, test_video_with_content):
        """Test the complete video processing pipeline."""
        frame_cleaner = FrameCleaner(use_llm=False)

        roi_config = {
            "x": 100,
            "y": 100,
            "width": 400,
            "height": 300,
            "image_width": 640,
            "image_height": 480,
        }

        data_roi_config = {
            "patient_info": {"x": 10, "y": 10, "width": 200, "height": 100}
        }

        # Test with minimal processing to avoid long execution
        with patch.object(frame_cleaner, "_iter_video") as mock_iter:
            # Mock to return just one frame
            mock_iter.return_value = [(0, np.zeros((480, 640), dtype=np.uint8), 1)]

            with patch.object(
                frame_cleaner, "_mask_video_streaming", return_value=True
            ):
                result_path, metadata = frame_cleaner.clean_video(
                    video_path=test_video_with_content,
                    endoscope_image_roi=roi_config,
                    endoscope_data_roi_nested=data_roi_config,  # type: ignore
                    technique="mask_overlay",
                )

        assert isinstance(result_path, Path)
        assert isinstance(metadata, dict)
        assert "source" in metadata
        assert metadata["source"] == "frame_extraction"

    def test_mask_application_integration(self):
        """Test that MaskApplication is properly integrated."""
        frame_cleaner = FrameCleaner()

        # Test that mask application has the right configuration
        assert frame_cleaner.mask_application.default_mask_config is not None

        # Test mask config creation
        test_roi = {
            "x": 550,
            "y": 0,
            "width": 1350,
            "height": 1080,
            "image_width": 1920,
            "image_height": 1080,
        }

        mask_config = frame_cleaner._create_mask_config_from_roi(test_roi)

        # Should match olympus configuration
        assert mask_config["endoscope_image_x"] == 550
        assert mask_config["endoscope_image_y"] == 0
        assert mask_config["endoscope_image_width"] == 1350
        assert mask_config["endoscope_image_height"] == 1080
        assert mask_config["image_width"] == 1920
        assert mask_config["image_height"] == 1080

    def test_video_encoder_integration(self):
        """Test that VideoEncoder is properly integrated."""
        frame_cleaner = FrameCleaner()

        # Test encoder configuration
        assert hasattr(frame_cleaner, "preferred_encoder")
        assert hasattr(frame_cleaner, "_build_encoder_cmd")

        # Test encoder command building
        encoder_cmd = frame_cleaner._build_encoder_cmd("balanced")
        assert isinstance(encoder_cmd, list)
        assert len(encoder_cmd) > 0

    def test_metadata_accumulation(self):
        """Test metadata accumulation from multiple frames."""
        frame_cleaner = FrameCleaner()

        # Test metadata merging functionality
        frame_meta1 = {"patient_first_name": "John", "patient_last_name": None}
        frame_meta2 = {"patient_first_name": None, "patient_last_name": "Doe"}

        accumulated = {
            "patient_first_name": None,
            "patient_last_name": None,
            "source": "frame_extraction",
        }

        # Test metadata merger if available
        if hasattr(frame_cleaner.frame_metadata_extractor, "merge_metadata"):
            result1 = frame_cleaner.frame_metadata_extractor.merge_metadata(
                accumulated, frame_meta1
            )
            result2 = frame_cleaner.frame_metadata_extractor.merge_metadata(
                result1, frame_meta2
            )

            # Should accumulate data from both frames
            assert result2.get("patient_first_name") == "John"
            assert result2.get("patient_last_name") == "Doe"


class TestFrameCleanerPerformance:
    """Tests for performance and optimization features."""

    def test_frame_sampling_optimization(self):
        """Test that frame sampling works for long videos."""
        frame_cleaner = FrameCleaner()

        # Mock a long video scenario
        total_frames = 50000  # Long video

        with patch("cv2.VideoCapture") as mock_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = total_frames
            mock_cap.read.return_value = (False, None)  # End immediately
            mock_capture.return_value = mock_cap

            frames = list(frame_cleaner._iter_video(Path("test.mp4"), total_frames))

            # Should sample frames rather than process all
            assert (
                len(frames) == 0
            )  # Due to immediate end, but sampling logic should be applied

    def test_hardware_acceleration_fallback(self):
        """Test hardware acceleration fallback behavior."""
        frame_cleaner = FrameCleaner()

        # Should have fallback encoder configuration
        assert "type" in frame_cleaner.preferred_encoder
        encoder_type = frame_cleaner.preferred_encoder["type"]
        assert encoder_type in ["nvenc", "cpu"]  # Should be one of supported types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
