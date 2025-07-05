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
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import cv2
import numpy as np
import pytesseract
from PIL import Image

from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.utils.ollama import ensure_ollama

logger = logging.getLogger(__name__)

class FrameCleaner:
    """
    FrameCleaner class for handling video frame extraction and sensitive data detection.
    
    This class provides methods to extract frames from a video, detect sensitive information
    using OCR, and re-encode the video without sensitive frames.
    """
    
    def __init__(self):
        ollama_proc = ensure_ollama()
        pass 
    
        

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
    
    def _detect_sensitive_meta_llm(
        self,
        ocr_text: str,
        report_reader: ReportReader,
        llm_model: str = "deepseek"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect sensitive metadata using LLM-powered extraction from ReportReader.
        
        Args:
            ocr_text: OCR-extracted text from frame
            report_reader: ReportReader instance with LLM extractors
            llm_model: LLM model to use ('deepseek', 'medllama', 'llama3')
            
        Returns:
            Tuple of (has_sensitive_data, metadata_dict)
        """
        try:
            # Call ReportReader's process_report with LLM extractor
            _, _, meta = report_reader.process_report(
                text=ocr_text,
                pdf_path=None,
                image_path=None,
                use_llm_extractor=llm_model,
                verbose=False
            )
            
            # Check for sensitive information in extracted metadata
            sensitive_fields = ['patient_first_name', 'patient_last_name', 'casenumber']
            has_sensitive_data = False
            
            # Check standard sensitive fields
            for field in sensitive_fields:
                value = meta.get(field)
                if value not in [None, '', 'Unknown']:
                    has_sensitive_data = True
                    break
            
            # Check for non-empty patient_dob
            if not has_sensitive_data:
                patient_dob = meta.get('patient_dob')
                if patient_dob and patient_dob.strip():
                    has_sensitive_data = True
            
            # Check for patient_gender that is not None
            if not has_sensitive_data:
                patient_gender = meta.get('patient_gender')
                if patient_gender is not None:
                    has_sensitive_data = True
            
            return has_sensitive_data, meta
            
        except Exception as e:
            logger.error(f"LLM metadata extraction failed: {e}")
            return False, {}

    def detect_sensitive_on_frame_extended(self, frame_path: Path, report_reader) -> bool:
        """
        Detect if a frame contains sensitive information using OCR + LLM-powered metadata extraction.
        
        Args:
            frame_path: Path to frame image
            report_reader: ReportReader instance with LLM extractors
            
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
            
            # Use LLM-powered metadata extraction
            has_sensitive, meta = self._detect_sensitive_meta_llm(
                ocr_text, report_reader, llm_model="deepseek"
            )
            
            if has_sensitive:
                logger.warning(f"LLM detected sensitive data in frame {frame_path.name}: {meta}")
                return True
            
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
            
            logger.info(f"Removing {len(frames_to_remove)} frames from video: {frames_to_remove}")
            
            # Create properly escaped filter expression for multiple frames
            # Escape commas in eq() expressions and join with + for OR logic
            idx_list = '+'.join([f'eq(n\\,{idx})' for idx in frames_to_remove])
            
            # Build video filter: select frames NOT in the removal list
            vf = f"select='not({idx_list})',setpts=N/FRAME_RATE/TB"
            
            # Build audio filter: keep audio in sync (or skip if no audio needed)
            af = f"aselect='not({idx_list})',asetpts=N/SR/TB"
            
            # Build ffmpeg command with properly quoted filters
            cmd = [
                'ffmpeg', '-i', str(original_video),
                '-vf', vf,
                '-af', af,
                '-y',  # Overwrite existing files
                str(output_video)
            ]
            
            logger.info(f"Re-encoding video without {len(frames_to_remove)} frames")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"ffmpeg re-encode output: {result.stderr}")
            
            if output_video.exists() and output_video.stat().st_size > 0:
                logger.info(f"Successfully created cleaned video: {output_video}")
                return True
            else:
                logger.error("Output video is empty or missing")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg re-encoding failed: {e.stderr}")
            # Fallback: try without audio filter if audio processing failed
            try:
                logger.warning("Retrying without audio processing...")
                cmd_no_audio = [
                    'ffmpeg', '-i', str(original_video),
                    '-vf', vf,
                    '-an',  # No audio
                    '-y',
                    str(output_video)
                ]
                result = subprocess.run(cmd_no_audio, capture_output=True, text=True, check=True)
                logger.info("Successfully re-encoded video without audio")
                return True
            except subprocess.CalledProcessError as e2:
                logger.error(f"ffmpeg re-encoding failed even without audio: {e2.stderr}")
                return False
        except Exception as e:
            logger.error(f"Video re-encoding error: {e}")
            return False
        
    def _load_mask(self, device_name: str) -> Dict[str, Any]:
        masks_dir  = Path(__file__).parent / "masks"
        mask_file  = masks_dir / f"{device_name}_mask.json"
        stub       = {
            "image_width": 1920,
            "image_height": 1080,
            "endoscope_image_x": 550,
            "endoscope_image_y": 0,
            "endoscope_image_width": 1350,
            "endoscope_image_height": 1080,
            "description": f"Mask configuration for {device_name}"
        }

        try:
            with mask_file.open() as f:
                return json.load(f)           # works if file is valid
        except (FileNotFoundError, json.JSONDecodeError):
            # create or overwrite with a fresh stub
            masks_dir.mkdir(parents=True, exist_ok=True)
            with mask_file.open("w") as f:
                json.dump(stub, f, indent=2)
            logger.warning(
                "Created or repaired mask file %s – please verify coordinates.",
                mask_file
            )
            return stub

                
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load/create mask configuration for {device_name}: {e}")
            raise FileNotFoundError(f"Could not load or create mask configuration for {device_name}: {e}")

    def _mask_video(self, input_video: Path, mask_config: Dict[str, Any], output_video: Path) -> bool:
        """
        Apply mask to video using FFmpeg to hide sensitive areas.
        
        Args:
            input_video: Path to input video file
            mask_config: Dictionary containing mask coordinates
            output_video: Path for output masked video
            
        Returns:
            True if masking succeeded, False otherwise
        """
        try:
            endoscope_x = mask_config.get("endoscope_image_x", 0)
            endoscope_y = mask_config.get("endoscope_image_y", 0)
            endoscope_w = mask_config.get("endoscope_image_width", 640)
            endoscope_h = mask_config.get("endoscope_image_height", 480)
            
            # Check if we can use simple crop (left strip masking)
            if endoscope_y == 0 and endoscope_h == mask_config.get("image_height", 1080):
                # Simple left crop case - crop everything to the right of the endoscope area
                crop_filter = f"crop=in_w-{endoscope_x}:in_h:{endoscope_x}:0"
                cmd = [
                    'ffmpeg', '-i', str(input_video),
                    '-vf', crop_filter,
                    '-c:a', 'copy',  # Preserve audio
                    '-y',
                    str(output_video)
                ]
                logger.info(f"Using simple crop mask: {crop_filter}")
            else:
                # Complex masking using drawbox to black out sensitive areas
                # Mask everything except the endoscope area
                mask_filters = []
                
                # Left rectangle (0 to endoscope_x)
                if endoscope_x > 0:
                    mask_filters.append(f"drawbox=0:0:{endoscope_x}:{mask_config.get('image_height', 1080)}:color=black@1:t=fill")
                
                # Right rectangle (endoscope_x + endoscope_w to image_width)
                right_start = endoscope_x + endoscope_w
                image_width = mask_config.get('image_width', 1920)
                if right_start < image_width:
                    right_width = image_width - right_start
                    mask_filters.append(f"drawbox={right_start}:0:{right_width}:{mask_config.get('image_height', 1080)}:color=black@1:t=fill")
                
                # Top rectangle (within endoscope x range, 0 to endoscope_y)
                if endoscope_y > 0:
                    mask_filters.append(f"drawbox={endoscope_x}:0:{endoscope_w}:{endoscope_y}:color=black@1:t=fill")
                
                # Bottom rectangle (within endoscope x range, endoscope_y + endoscope_h to image_height)
                bottom_start = endoscope_y + endoscope_h
                image_height = mask_config.get('image_height', 1080)
                if bottom_start < image_height:
                    bottom_height = image_height - bottom_start
                    mask_filters.append(f"drawbox={endoscope_x}:{bottom_start}:{endoscope_w}:{bottom_height}:color=black@1:t=fill")
                
                # Combine all mask filters
                vf = ','.join(mask_filters)
                
                cmd = [
                    'ffmpeg', '-i', str(input_video),
                    '-vf', vf,
                    '-c:a', 'copy',  # Preserve audio
                    '-y',
                    str(output_video)
                ]
                logger.info(f"Using complex drawbox mask with {len(mask_filters)} regions")
            
            logger.info(f"Applying mask to video: {input_video} -> {output_video}")
            logger.debug(f"FFmpeg masking command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"FFmpeg masking output: {result.stderr}")
            
            if output_video.exists() and output_video.stat().st_size > 0:
                logger.info(f"Successfully created masked video: {output_video}")
                return True
            else:
                logger.error("Masked video is empty or missing")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg masking failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Video masking error: {e}")
            return False

    def clean_video(
        self,
        video_path: Path,
        report_reader=ReportReader(),
        tmp_dir: Optional[Path] = None,
        device_name: Optional[str] = None
    ) -> Path:
        """
        Clean video by removing frames with sensitive information or masking persistent overlays.
        
        Args:
            video_path: Path to input video
            report_reader: ReportReader instance for sensitive data detection
            tmp_dir: Temporary directory for processing (optional)
            device_name: Name of endoscopy device for mask configuration (optional)
            
        Returns:
            Path to cleaned video file (with "_anony" suffix)
            
        Raises:
            RuntimeError: If video processing fails
        """
        if tmp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix='frame_cleaner_'))
        
        # Default device name if not provided
        if device_name is None:
            device_name = "generic"
        
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
            
            if not sensitive_frame_indices:
                for i, frame_path in enumerate(frame_paths):
                    if self.detect_sensitive_on_frame_extended(frame_path, report_reader):
                        sensitive_frame_indices.append(i)
            
            total_frames = len(frame_paths)
            sensitive_ratio = len(sensitive_frame_indices) / total_frames if total_frames > 0 else 0
            
            logger.info(f"Found {len(sensitive_frame_indices)} sensitive frames out of {total_frames} "
                       f"({sensitive_ratio:.1%} ratio)")
            
            # Decision: mask vs frame removal based on 10% threshold
            if sensitive_ratio > 0.10:
                logger.info(f"Sensitive content ratio ({sensitive_ratio:.1%}) exceeds 10% threshold. "
                           f"Applying mask instead of removing frames.")
                try:
                    mask_config = self._load_mask(device_name)
                    success = self._mask_video(video_path, mask_config, output_video)
                    
                    if not success:
                        logger.error("Failed to create masked video, using original")
                        import shutil
                        shutil.copy2(video_path, output_video)
                        
                except Exception as e:
                    logger.error(f"Masking failed: {e}. Falling back to frame removal.")
                    # Fall back to frame removal if masking fails
                    success = self.remove_frames_from_video(
                        video_path, 
                        sensitive_frame_indices, 
                        output_video,
                        total_frames=total_frames
                    )
                    
                    if not success:
                        logger.error("Failed to create cleaned video, using original")
                        import shutil
                        shutil.copy2(video_path, output_video)
            else:
                # Traditional frame removal for videos with < 10% sensitive content
                if not sensitive_frame_indices:
                    logger.info("No sensitive frames detected, copying original video")
                    import shutil
                    shutil.copy2(video_path, output_video)
                    return output_video
                
                logger.info(f"Sensitive content ratio ({sensitive_ratio:.1%}) below 10% threshold. "
                           f"Removing {len(sensitive_frame_indices)} sensitive frames.")
                
                # Re-encode video without sensitive frames
                success = self.remove_frames_from_video(
                    video_path, 
                    sensitive_frame_indices, 
                    output_video,
                    total_frames=total_frames
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