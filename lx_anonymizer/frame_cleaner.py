"""
Frame-level anonymization module for video processing.

This module provides functionality to:
- Extract frames from videos using ffmpeg
- Apply OCR to detect sensitive information (names, DOB, case numbers)
- Remove or mask frames containing sensitive data
- Re-encode cleaned videos

Uses the same spaCy + regex logic from lx_anonymizer.report_reader for consistency.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from lx_anonymizer.report_reader import ReportReader
import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

class FrameCleaner:
    """
    FrameCleaner class for handling video frame extraction and sensitive data detection.
    
    This class provides methods to extract frames from a video, detect sensitive information
    using OCR, and re-encode the video without sensitive frames.
    """
    
    def __init__(self):
        pass  # Initialization logic if needed in the future
    
        

    def extract_frames(self, video_path: Path, output_dir: Path, max_frames: Optional[int] = None) -> List[Path]:
        """
        Extract frames from video using ffmpeg.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            max_frames: Maximum number of frames to extract (None for all)
            
        Returns:
            List of paths to extracted frame images
            
        Raises:
            RuntimeError: If ffmpeg extraction fails
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', 'fps=1',  # Extract 1 frame per second (adjust as needed)
            '-y',  # Overwrite existing files
            str(output_dir / 'frame_%04d.jpg')
        ]
        
        # Limit frames if specified
        if max_frames:
            cmd.insert(-1, '-frames:v')
            cmd.insert(-1, str(max_frames))
        
        try:
            logger.info(f"Extracting frames from {video_path} to {output_dir}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"ffmpeg output: {result.stderr}")
            
            # Get list of created frame files
            frame_files = sorted(output_dir.glob('frame_*.jpg'))
            logger.info(f"Extracted {len(frame_files)} frames")
            return frame_files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg frame extraction failed: {e.stderr}")
            raise RuntimeError(f"Frame extraction failed: {e}")


    def detect_sensitive_on_frame(self, frame_path: Path, report_reader) -> bool:
        """
        Detect if a frame contains sensitive information using OCR + name detection.
        
        Args:
            frame_path: Path to frame image
            report_reader: ReportReader instance with PatientDataExtractor
            
        Returns:
            True if frame contains sensitive data, False otherwise
        """
        try:
            # Load image and apply OCR
            image = Image.open(frame_path)
            
            # Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            # Apply OCR
            ocr_text = pytesseract.image_to_string(image, lang='deu')
            
            if not ocr_text.strip():
                logger.debug(f"No text detected in frame {frame_path.name}")
                return False
            
            logger.debug(f"OCR text from {frame_path.name}: {ocr_text[:100]}...")
            
            # Use the same name detection logic as report_reader
            patient_info = report_reader.patient_extractor(ocr_text)
            
            # Check if sensitive information was found
            sensitive_fields = ['patient_first_name', 'patient_last_name', 'casenumber']
            has_sensitive_data = any(
                patient_info.get(field) not in [None, '', 'Unknown'] 
                for field in sensitive_fields
            )
            
            if has_sensitive_data:
                logger.warning(f"Sensitive data detected in frame {frame_path.name}: {patient_info}")
                return True
            
            # TODO: Add additional checks for DOB patterns, case numbers, etc.
            # For now, rely on PatientDataExtractor's comprehensive detection
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_path}: {e}")
            # Fail-safe: if OCR crashes, keep the frame (better none deleted than all lost)
            return False


    def remove_frames_from_video(
        self,
        original_video: Path, 
        frames_to_remove: List[int], 
        output_video: Path,
        total_frames: Optional[int] = None
    ) -> bool:
        """
        Re-encode video without specified frames.
        
        Args:
            original_video: Path to original video
            frames_to_remove: List of frame numbers to remove (0-based)
            output_video: Path for output video
            total_frames: Total frame count (for optimization)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not frames_to_remove:
                logger.info("No frames to remove, copying original video")
                import shutil
                shutil.copy2(original_video, output_video)
                return True
            
            # Create temporary filter file for ffmpeg
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as filter_file:
                # Build filter to skip specific frames
                # For simplicity, we'll use a frame selection approach
                # TODO: Optimize this for large videos with many frames to remove
                
                if total_frames:
                    # Create list of frames to keep
                    frames_to_keep = [i for i in range(total_frames) if i not in frames_to_remove]
                    
                    # Write frame selection filter
                    select_expr = '+'.join([f'eq(n,{frame})' for frame in frames_to_keep[:100]])  # Limit for performance
                    if len(frames_to_keep) > 100:
                        logger.warning(f"Too many frames to process efficiently ({len(frames_to_keep)}), using simplified approach")
                        select_expr = f'not(eq(n,{frames_to_remove[0]}))'  # Just remove first problematic frame
                    
                    filter_file.write(f"select='{select_expr}',setpts=N/FRAME_RATE/TB")
                else:
                    # Fallback: just remove first problematic frame
                    filter_file.write(f"select='not(eq(n,{frames_to_remove[0]}))',setpts=N/FRAME_RATE/TB")
                
                filter_path = filter_file.name
            
            try:
                # Build ffmpeg command with frame filtering
                cmd = [
                    'ffmpeg', '-i', str(original_video),
                    '-vf', f'select=not(eq(n,{frames_to_remove[0]})),setpts=N/FRAME_RATE/TB',
                    '-af', 'aselect=concatdec_select,asetpts=N/SR/TB',  # Sync audio
                    '-y',
                    str(output_video)
                ]
                
                logger.info(f"Re-encoding video without {len(frames_to_remove)} frames")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.debug(f"ffmpeg re-encode output: {result.stderr}")
                
                if output_video.exists() and output_video.stat().st_size > 0:
                    logger.info(f"Successfully created cleaned video: {output_video}")
                    return True
                else:
                    logger.error("Output video is empty or missing")
                    return False
                    
            finally:
                # Clean up filter file
                Path(filter_path).unlink(missing_ok=True)
                
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg re-encoding failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Video re-encoding error: {e}")
            return False


    def clean_video(
        self,
        video_path: Path,
        report_reader=ReportReader(),
        tmp_dir: Optional[Path] = None
    ) -> Path:
        """
        Clean video by removing frames with sensitive information.
        
        Args:
            video_path: Path to input video
            report_reader: ReportReader instance for sensitive data detection
            tmp_dir: Temporary directory for processing (optional)
            
        Returns:
            Path to cleaned video file (with "_anony" suffix)
            
        Raises:
            RuntimeError: If video processing fails
        """
        if tmp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix='frame_cleaner_'))
        
        try:
            # Create output path with _anony suffix
            output_video = video_path.with_stem(f"{video_path.stem}_anony")
            
            # Extract frames for analysis
            frames_dir = tmp_dir / 'frames'
            frame_paths = self.extract_frames(video_path, frames_dir, max_frames=100)  # Limit for performance
            
            if not frame_paths:
                logger.warning("No frames extracted, copying original video")
                import shutil
                shutil.copy2(video_path, output_video)
                return output_video
            
            # Analyze frames for sensitive content
            sensitive_frame_indices = []
            for i, frame_path in enumerate(frame_paths):
                if self.detect_sensitive_on_frame(frame_path, report_reader):
                    sensitive_frame_indices.append(i)
            
            logger.info(f"Found {len(sensitive_frame_indices)} sensitive frames out of {len(frame_paths)}")
            
            # Re-encode video without sensitive frames
            success = self.remove_frames_from_video(
                video_path, 
                sensitive_frame_indices, 
                output_video,
                total_frames=len(frame_paths)
            )
            
            if not success:
                logger.error("Failed to create cleaned video, using original")
                import shutil
                shutil.copy2(video_path, output_video)
            
            return output_video
            
        except Exception as e:
            logger.error(f"Video cleaning failed: {e}")
            # Fail-safe: copy original if cleaning fails
            output_video = video_path.with_stem(f"{video_path.stem}_anony")
            import shutil
            shutil.copy2(video_path, output_video)
            return output_video
            
        finally:
            # Clean up temporary files
            if tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)