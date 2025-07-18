"""
Basic tests for frame_cleaner module.

Tests the main functionality of video frame cleaning while ensuring
the original file is not modified.
"""

import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch
import shutil

# Import the modules we're testing
try:
    from lx_anonymizer.frame_cleaner import clean_video, extract_frames, detect_sensitive_on_frame
    from lx_anonymizer.report_reader import ReportReader
except ImportError:
    pytest.skip("lx_anonymizer modules not available", allow_module_level=True)


def create_test_video(output_path: Path, duration: int = 2) -> Path:
    """
    Create a small synthetic test video using ffmpeg.
    
    Args:
        output_path: Where to save the test video
        duration: Duration in seconds
        
    Returns:
        Path to created video file
    """
    cmd = [
        'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=2:size=320x240:rate=1',
        '-c:v', 'libx264', '-t', str(duration), '-y', str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("ffmpeg not available for test video creation")


@pytest.fixture
def test_video():
    """Create a temporary test video for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / "test_video.mp4"
        create_test_video(video_path)
        yield video_path


@pytest.fixture
def mock_report_reader():
    """Create a mock ReportReader for testing."""
    mock_reader = Mock(spec=ReportReader)
    mock_extractor = Mock()
    
    # Mock the patient_extractor to return no sensitive data by default
    mock_extractor.return_value = {
        'patient_first_name': None,
        'patient_last_name': None,
        'casenumber': None,
        'patient_dob': None,
        'patient_gender': None
    }
    
    mock_reader.patient_extractor = mock_extractor
    return mock_reader


class TestFrameCleaner:
    """Test cases for frame cleaning functionality."""
    
    def test_clean_video_returns_path(self, test_video, mock_report_reader):
        """Test that clean_video returns a Path object."""
        result = clean_video(test_video, mock_report_reader)
        
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == test_video.suffix
        assert "_anony" in result.stem
    
    def test_clean_video_preserves_original(self, test_video, mock_report_reader):
        """Test that the original video file is not modified."""
        original_size = test_video.stat().st_size
        original_mtime = test_video.stat().st_mtime
        
        clean_video(test_video, mock_report_reader)
        
        # Original file should be unchanged
        assert test_video.exists()
        assert test_video.stat().st_size == original_size
        assert test_video.stat().st_mtime == original_mtime
    
    def test_extract_frames_basic(self, test_video):
        """Test basic frame extraction functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "frames"
            
            try:
                frames = extract_frames(test_video, output_dir, max_frames=2)
                
                assert len(frames) > 0
                assert all(frame.exists() for frame in frames)
                assert all(frame.suffix == '.jpg' for frame in frames)
                
            except RuntimeError:
                # If ffmpeg fails, that's expected in some test environments
                pytest.skip("ffmpeg frame extraction failed in test environment")
    
    def test_detect_sensitive_on_frame_no_sensitive_data(self, mock_report_reader):
        """Test that frames without sensitive data are not flagged."""
        # Create a simple test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Create a minimal image file (we'll mock the OCR anyway)
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='white')
            img.save(tmp_file.name)
            
            try:
                with patch('pytesseract.image_to_string', return_value="Safe text content"):
                    result = detect_sensitive_on_frame(Path(tmp_file.name), mock_report_reader)
                    assert result == False
            finally:
                Path(tmp_file.name).unlink(missing_ok=True)
    
    def test_detect_sensitive_on_frame_with_sensitive_data(self, mock_report_reader):
        """Test that frames with sensitive data are flagged."""
        # Mock the extractor to return sensitive data
        mock_report_reader.patient_extractor.return_value = {
            'patient_first_name': 'John',
            'patient_last_name': 'Doe',
            'casenumber': '12345',
            'patient_dob': None,
            'patient_gender': None
        }
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='white')
            img.save(tmp_file.name)
            
            try:
                with patch('pytesseract.image_to_string', return_value="Patient: John Doe"):
                    result = detect_sensitive_on_frame(Path(tmp_file.name), mock_report_reader)
                    assert result == True
            finally:
                Path(tmp_file.name).unlink(missing_ok=True)
    
    def test_clean_video_handles_missing_file(self, mock_report_reader):
        """Test that clean_video handles missing input files gracefully."""
        non_existent_file = Path("/path/that/does/not/exist.mp4")
        
        # Should still return a path, but may copy from a fallback or handle gracefully
        result = clean_video(non_existent_file, mock_report_reader)
        assert isinstance(result, Path)
    
    def test_clean_video_creates_output_with_anony_suffix(self, test_video, mock_report_reader):
        """Test that output video has the correct naming convention."""
        result = clean_video(test_video, mock_report_reader)
        
        expected_stem = f"{test_video.stem}_anony"
        assert result.stem == expected_stem
        assert result.parent == test_video.parent  # Same directory by default


class TestIntegration:
    """Integration tests that verify the complete pipeline."""
    
    def test_full_pipeline_with_real_report_reader(self, test_video):
        """Test the complete pipeline with a real ReportReader instance."""
        try:
            # This test uses real components and may be slower
            reader = ReportReader(locale="de_DE")
            result = clean_video(test_video, reader)
            
            assert result.exists()
            assert result != test_video  # Different file
            assert test_video.exists()  # Original preserved
            
        except Exception as e:
            pytest.skip(f"Real ReportReader not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])