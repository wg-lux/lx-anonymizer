"""
Frame-level anonymization module for video processing.

This module provides functionality to:
- Extract frames from videos using ffmpeg
- Apply specialized frame OCR to detect sensitive information
- Remove or mask frames containing sensitive data
- Re-encode cleaned videos

Uses specialized frame processing components separated from PDF logic.
"""

import logging
import subprocess
import tempfile
import json
import os
from pathlib import Path
from tkinter import N
from typing import List, Optional, Tuple, Dict, Any, Union
import cv2
import numpy as np
from PIL import Image
import pytesseract

from lx_anonymizer.frame_ocr import FrameOCR
from lx_anonymizer.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.best_frame_text import BestFrameText
from lx_anonymizer.utils.ollama import ensure_ollama
from lx_anonymizer.ollama_llm_meta_extraction import extract_with_fallback
from lx_anonymizer.report_reader import ReportReader

logger = logging.getLogger(__name__)

class FrameCleaner:
    """
    FrameCleaner class for handling video frame extraction and sensitive data detection.
    
    This class provides methods to extract frames from a video, detect sensitive information
    using specialized frame OCR, and re-encode the video without sensitive frames.
    """
    
    def __init__(self):
        # Initialize specialized frame processing components
        self.frame_ocr = FrameOCR()
        self.frame_metadata_extractor = FrameMetadataExtractor()
        self.best_frame_text = BestFrameText()
        
        # Initialize Ollama for LLM processing
        self.ollama_proc = ensure_ollama()
    
    def clean_video(
        self,
        video_path: Path,
        video_file_obj=None,  # Add VideoFile object to store metadata
        report_reader=None,  # Keep for compatibility but use frame-specific logic
        tmp_dir: Optional[Path] = None,
        device_name: Optional[str] = None,
        endoscope_roi: Optional[Dict[str, Any]] = None,
        processor_rois: Optional[Dict[str, Dict[str, Any]]] = None,
        frame_paths: Optional[list[Path]] = None
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Clean video by removing frames with sensitive information or masking persistent overlays.
        
        Args:
            video_path: Path to input video
            video_file_obj: VideoFile Django model instance to store extracted metadata
            report_reader: Kept for compatibility (now uses frame-specific processing)
            tmp_dir: Temporary directory for processing (optional)
            device_name: Name of endoscopy device for mask configuration (optional)
            endoscope_roi: Endoscope ROI from processor for masking (optional)
            processor_rois: All processor ROIs for comprehensive anonymization (optional)
            
        Returns:
            Tuple of (Path to cleaned video file, extracted_metadata_dict)
            
        Raises:
            RuntimeError: If video processing fails
        """
        if tmp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix='frame_cleaner_'))
        
        # Default device name if not provided
        if device_name is None:
            device_name = "olympus_cv_1500"
        
        # Create output video path
        output_video = video_path.with_stem(f"{video_path.stem}_anony")
        
        # Accumulate metadata from all frames using specialized extractor
        accumulated_metadata = {
            "patient_first_name": None,
            "patient_last_name": None,
            "patient_dob": None,
            "casenumber": None,
            "patient_gender": None,
            "examination_date": None,
            "examination_time": None,
            "examiner": None,
            "representative_ocr_text": None,
            "source": "frame_extraction"
        }

        try:
            # ▼ Adaptive streaming + specialized frame OCR
            sensitive_idx: list[int] = []
            
            # Get total frames once
            total_frames = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT))
            skip = 1

            for abs_i, gray_frame, skip in self._iter_video(video_path, total_frames):
                # Use specialized frame OCR instead of basic pytesseract
                ocr_text, avg_conf, _ = self.frame_ocr.extract_text_from_frame(
                    gray_frame, 
                    roi=endoscope_roi,
                    high_quality=True
                )

                if not ocr_text:
                    continue
                
                # Feed the 'best text' sampler
                self.best_frame_text.push(ocr_text, avg_conf)
                
                if self.best_frame_text:
                    frame_metadata = self.extract_metadata_deepseek(ocr_text)
                    if frame_metadata == {}:
                        frame_metadata = self.frame_metadata_extractor.extract_metadata_from_frame_text(ocr_text)
                
                # Check if frame contains sensitive content
                has_sensitive = self.frame_metadata_extractor.is_sensitive_content(frame_metadata)
                
                # Accumulate non-null metadata from this frame
                if frame_metadata:
                    accumulated_metadata = self.frame_metadata_extractor.merge_metadata(
                        accumulated_metadata, frame_metadata
                    )
                    if has_sensitive:
                        logger.info(f"Found sensitive data in frame {abs_i}: {frame_metadata}")

                # Mark frame as sensitive if it contains sensitive data
                if has_sensitive:
                    sensitive_idx.append(abs_i)
            
            # Get best representative text
            best_summary = self.best_frame_text.reduce()
            representative_text = best_summary.get("best", "")
            accumulated_metadata["representative_ocr_text"] = representative_text
            
            logger.info("Representative OCR text – best: %s", representative_text)
            
            # Try to extract additional metadata from the best OCR text using LLM if available
            if representative_text and report_reader:
                try:
                    has_llm_sensitive, llm_metadata = self._detect_sensitive_meta_llm(
                        representative_text, report_reader, llm_model="deepseek"
                    )
                    
                    # Merge LLM metadata with accumulated metadata
                    if llm_metadata:
                        accumulated_metadata = self.frame_metadata_extractor.merge_metadata(
                            accumulated_metadata, llm_metadata
                        )
                        logger.info(f"LLM enhanced metadata: {llm_metadata}")
                        
                except Exception as e:
                    logger.warning(f"LLM metadata enhancement failed: {e}")
            
            # Store extracted metadata in VideoFile's SensitiveMeta
            if video_file_obj and accumulated_metadata:
                self._update_video_sensitive_meta(video_file_obj, accumulated_metadata)
            
            # Apply padding to sensitive frame indices
            if skip > 1:
                pad = skip - 1
                padded_idx = set()
                for f in sensitive_idx:
                    padded_idx.update(range(max(0, f - pad), min(total_frames, f + pad + 1)))
                sensitive_idx = sorted(padded_idx)
            
            sensitive_ratio = len(sensitive_idx) / total_frames if total_frames > 0 else 0
            
            logger.info(f"Found {len(sensitive_idx)} sensitive frames out of {total_frames} "
                       f"({sensitive_ratio:.1%} ratio)")
            
            # Decision: mask vs frame removal based on 10% threshold
            if sensitive_ratio > 0.10:
                logger.info(f"Sensitive content ratio ({sensitive_ratio:.1%}) exceeds 10% threshold. "
                           f"Applying mask instead of removing frames.")
                try:
                    # Use processor ROI if available, otherwise fall back to device mask
                    if endoscope_roi and self._validate_roi(endoscope_roi):
                        logger.info("Using processor endoscope ROI for masking")
                        mask_config = self._create_mask_config_from_roi(endoscope_roi, processor_rois)
                    else:
                        logger.info("Using device-specific mask configuration")
                        mask_config = self._load_mask(device_name)
                    
                    success = self._mask_video(video_path, mask_config, output_video)
                    
                    if not success:
                        logger.error("Failed to create masked video, using original")
                        import shutil
                        shutil.copy2(video_path, output_video)
                        
                except Exception as e:
                    logger.exception(f"Masking failed: {e}. Falling back to frame removal.")
                    # Fall back to frame removal if masking fails
                    success = self.remove_frames_from_video(
                        video_path, 
                        sensitive_idx, 
                        output_video,
                        total_frames=total_frames
                    )
                    
                    if not success:
                        logger.error("Failed to create cleaned video, using original")
                        import shutil
                        shutil.copy2(video_path, output_video)
            else:
                # Traditional frame removal for videos with < 10% sensitive content
                if not sensitive_idx:
                    logger.info("No sensitive frames detected, copying original video")
                    import shutil
                    shutil.copy2(video_path, output_video)
                    return output_video, accumulated_metadata
                
                logger.info(f"Sensitive content ratio ({sensitive_ratio:.1%}) below 10% threshold. "
                           f"Removing {len(sensitive_idx)} sensitive frames.")
                
                # Re-encode video without sensitive frames
                success = self.remove_frames_from_video(
                    video_path, 
                    sensitive_idx, 
                    output_video,
                    total_frames=total_frames
                )
                
                if not success:
                    logger.error("Failed to create cleaned video, using original")
                    import shutil
                    shutil.copy2(video_path, output_video)
            
            # Verify output exists
            if not output_video.exists():
                logger.warning("Output video does not exist, creating fallback copy")
                import shutil
                shutil.copy2(video_path, output_video)
            
            return output_video, accumulated_metadata
            
        except Exception as e:
            logger.exception(f"Video cleaning failed: {e}")
            # Fail-safe: copy original if cleaning fails
            import shutil
            shutil.copy2(video_path, output_video)
            return output_video, accumulated_metadata
            
        finally:
            # Clean up temporary files
            if tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)


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
                return True, meta
            
            return False, _
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_path}: {e}")
            # Fail-safe: if OCR crashes, keep the frame (better none deleted than all lost)
            return False
        
    def detect_sensitive_on_frame_text(
        self, ocr_text: str, report_reader: ReportReader) -> bool:
        """
        Detect if a frame text contains sensitive information using spaCy + regex.
        Args:
            ocr_text: OCR-extracted text from frame
            report_reader: ReportReader instance with PatientDataExtractor
        Returns:
            True if frame contains sensitive data, False otherwise
        """
        # Use LLM-powered metadata extraction
        has_sensitive, meta = self._detect_sensitive_meta_llm(
            ocr_text, report_reader, llm_model="deepseek"
        )
        
        if has_sensitive:
            logger.warning(f"LLM detected sensitive data in text: {meta}")
            return True, meta
        else:            
            logger.debug("No sensitive data detected in text using LLM.")
            return False, None

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
    
    def _validate_roi(self, roi: Dict[str, Any]) -> bool:
        """
        Validate that ROI dictionary contains required fields and reasonable values.
        
        Args:
            roi: ROI dictionary with x, y, width, height keys
            
        Returns:
            True if ROI is valid, False otherwise
        """
        if not isinstance(roi, dict):
            return False
        
        required_keys = ['x', 'y', 'width', 'height']
        if not all(key in roi for key in required_keys):
            logger.warning(f"ROI missing required keys. Expected: {required_keys}, got: {list(roi.keys())}")
            return False
        
        # Check for reasonable values (non-negative, not too large)
        try:
            x, y, width, height = roi['x'], roi['y'], roi['width'], roi['height']
            
            if any(val < 0 for val in [x, y, width, height]):
                logger.warning(f"ROI contains negative values: {roi}")
                return False
                
            if width == 0 or height == 0:
                logger.warning(f"ROI has zero width or height: {roi}")
                return False
                
            if any(val > 5000 for val in [x, y, width, height]):
                logger.warning(f"ROI values seem unreasonably large: {roi}")
                return False
                
            return True
            
        except (TypeError, ValueError) as e:
            logger.warning(f"ROI contains invalid values: {roi}, error: {e}")
            return False

    def _create_mask_config_from_roi(
        self, 
        endoscope_roi: Dict[str, Any], 
        processor_rois: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create mask configuration from processor ROI information.
        
        Args:
            endoscope_roi: Endoscope ROI from processor
            processor_rois: All processor ROIs (optional, for context)
            
        Returns:
            Mask configuration dictionary compatible with _mask_video
        """
        # Extract endoscope ROI coordinates
        endoscope_x = int(endoscope_roi.get('x', 0))
        endoscope_y = int(endoscope_roi.get('y', 0))
        endoscope_width = int(endoscope_roi.get('width', 640))
        endoscope_height = int(endoscope_roi.get('height', 480))
        
        # Estimate image dimensions (use common video resolutions as fallback)
        # In practice, these might come from video metadata or processor configuration
        image_width = 1920  # Default HD width
        image_height = 1080  # Default HD height
        
        # Try to infer image dimensions from ROI + some margin
        if endoscope_x + endoscope_width > image_width:
            image_width = endoscope_x + endoscope_width + 100  # Add some margin
        if endoscope_y + endoscope_height > image_height:
            image_height = endoscope_y + endoscope_height + 100  # Add some margin
        
        mask_config = {
            "image_width": image_width,
            "image_height": image_height,
            "endoscope_image_x": endoscope_x,
            "endoscope_image_y": endoscope_y,
            "endoscope_image_width": endoscope_width,
            "endoscope_image_height": endoscope_height,
            "description": f"Mask configuration created from processor ROI",
            "roi_source": "processor"
        }
        
        logger.info(f"Created mask config from processor ROI: "
                   f"endoscope=({endoscope_x},{endoscope_y},{endoscope_width},{endoscope_height}), "
                   f"image=({image_width},{image_height})")
        
        return mask_config
    
    def video_ocr_stream(self, frame_paths: List[Path]):
        """
        Yield (ocr_text, avg_confidence) for every frame in frame_paths.

        Confidence is the mean of Tesseract word-level confidences,
        normalised to [0,1]. Empty-text frames are skipped.
        """
        for fp in frame_paths:
            # load once, convert to L for better OCR accuracy
            img = Image.open(fp).convert('L')

            # word-level OCR with confidences
            data = pytesseract.image_to_data(
                img, lang='deu',
                output_type=pytesseract.Output.DICT
            )

            words = [w for w in data["text"] if w.strip()]
            if not words:
                continue                      # nothing recognisable

            # average confidence; Tesseract returns -1 for “no conf”
            confs = [
                int(c) for c in data["conf"]
                if c.isdigit() and int(c) >= 0
            ]
            avg_conf = (sum(confs) / len(confs) / 100) if confs else 0.0
            yield " ".join(words), avg_conf
            
    def _iter_video(self, video_path: Path, total_frames: int) -> tuple[int, np.ndarray, int]:
        """
        Yield (abs_frame_index, gray_frame, skip_value) with adaptive subsampling:
            <1 000  frames → every frame
            1 000-9 999     → every 3rd
            ≥10 000         → every 5th
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open %s", video_path)
            return

        skip = 1 if total_frames < 1_000 else 3 if total_frames < 10_000 else 5
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % skip == 0:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                yield idx, gray, skip
            idx += 1
        cap.release()

    def _update_video_sensitive_meta(self, video_file_obj, metadata: Dict[str, Any]):
        """
        Update VideoFile's SensitiveMeta with extracted metadata from frames.
        
        Args:
            video_file_obj: VideoFile Django model instance
            metadata: Extracted metadata dictionary
        """
        try:
            # Import here to avoid circular imports
            from endoreg_db.models import SensitiveMeta
            
            # Get or create SensitiveMeta for this video file
            sensitive_meta, created = SensitiveMeta.objects.get_or_create(
                video_file=video_file_obj,
                defaults={
                    'patient_first_name': metadata.get('patient_first_name'),
                    'patient_last_name': metadata.get('patient_last_name'),
                    'patient_dob': metadata.get('patient_dob'),
                    'casenumber': metadata.get('casenumber'),
                    'patient_gender': metadata.get('patient_gender'),
                    'examination_date': metadata.get('examination_date'),
                    'examination_time': metadata.get('examination_time'),
                    'examiner': metadata.get('examiner'),
                    'representative_ocr_text': metadata.get('representative_ocr_text', ''),
                }
            )
            
            # If not created, update existing with non-empty values
            if not created:
                updated = False
                for field in ['patient_first_name', 'patient_last_name', 'patient_dob', 
                             'casenumber', 'patient_gender', 'examination_date', 
                             'examination_time', 'examiner', 'representative_ocr_text']:
                    value = metadata.get(field)
                    if value and value not in [None, '', 'Unknown']:
                        current_value = getattr(sensitive_meta, field, None)
                        if not current_value or current_value in [None, '', 'Unknown']:
                            setattr(sensitive_meta, field, value)
                            updated = True
                
                if updated:
                    sensitive_meta.save()
            
            logger.info(f"{'Created' if created else 'Updated'} SensitiveMeta for video {video_file_obj.id}")
            
        except Exception as e:
            logger.error(f"Failed to update video sensitive metadata: {e}")

    def _safe_conf_list(self, raw_conf):
        """Convert values from data['conf'] to int >= 0 safely"""
        confs = []
        for c in raw_conf:
            try:
                conf_int = int(c)            # works for both str AND int
            except (TypeError, ValueError):
                continue
            if conf_int >= 0:
                confs.append(conf_int)
        return confs

    def extract_metadata_deepseek(self, text: str) -> Dict[str, Any]:
        """Extract metadata using DeepSeek via Ollama structured output."""
        logger.info("Attempting metadata extraction with DeepSeek (Ollama Structured Output)")
        # Use the wrapper that handles retries and returns {} on failure
        meta = extract_with_fallback(text, model="deepseek-r1:1.5b")
        if not meta:
            logger.warning("DeepSeek Ollama extraction failed, returning empty dict.")
        else:
            logger.info("DeepSeek Ollama extraction successful.")
        return meta

    def extract_metadata_medllama(self, text: str) -> Dict[str, Any]:
        """Extract metadata using MedLLaMA via Ollama structured output."""
        logger.info("Attempting metadata extraction with MedLLaMA (Ollama Structured Output)")
        # Use the wrapper that handles retries and returns {} on failure
        meta = extract_with_fallback(text, model="rjmalagon/medllama3-v20:fp16")
        if not meta:
            logger.warning("MedLLaMA Ollama extraction failed, returning empty dict.")
        else:
            logger.info("MedLLaMA Ollama extraction successful.")
        return meta

    def extract_metadata_llama3(self, text: str) -> Dict[str, Any]:
        """Extract metadata using Llama3 via Ollama structured output."""
        logger.info("Attempting metadata extraction with Llama3 (Ollama Structured Output)")
        # Use the wrapper that handles retries and returns {} on failure
        meta = extract_with_fallback(text, model="llama3:8b")  # Or llama3:70b if available/needed
        if not meta:
            logger.warning("Llama3 Ollama extraction failed, returning empty dict.")
        else:
            logger.info("Llama3 Ollama extraction successful.")
        return meta