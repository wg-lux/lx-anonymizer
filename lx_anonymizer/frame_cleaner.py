"""
Frame-level anonymization module for video processing.

This module provides functionality to:
- Extract frames from videos using ffmpeg
- Apply specialized frame OCR to detect sensitive information
- Remove or mask frames containing sensitive data
- Re-encode cleaned videos

Uses specialized frame processing components separated from PDF logic.
"""

import math
import logging
import subprocess
import tempfile
import json
import os
import stat
import time
import shutil

from pathlib import Path
from tkinter import N
from typing import List, Optional, Tuple, Dict, Any, Union, Iterator
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
from lx_anonymizer.minicpm_ocr import MiniCPMVisionOCR, create_minicpm_ocr, _can_load_model
from lx_anonymizer.spacy_extractor import PatientDataExtractor
from typing_inspection.typing_objects import NoneType

logger = logging.getLogger(__name__)


class FrameCleaner:
    """
    FrameCleaner class for handling video frame extraction and sensitive data detection.
    
    This class provides methods to extract frames from a video, detect sensitive information
    using specialized frame OCR (including MiniCPM-o 2.6), and re-encode the video without sensitive frames.
    
    New features:
    - Stream-based processing with FFmpeg -c copy to avoid full transcoding
    - Named pipe (FIFO) support for in-memory video streaming
    - Pixel format conversion optimization for minimal re-encoding
    """
    
    def __init__(self, use_minicpm: bool = True, minicpm_config: Optional[Dict[str, Any]] = None):
        # Initialize specialized frame processing components
        self.frame_ocr = FrameOCR()
        self.frame_metadata_extractor = FrameMetadataExtractor()
        self.PatientDataExtractor = PatientDataExtractor()
        self.best_frame_text = BestFrameText()
        
        # Initialize Ollama for LLM processing
        self.ollama_proc = ensure_ollama()
        
        # Initialize MiniCPM-o 2.6 if enabled
        self.use_minicpm = use_minicpm
        self.minicpm_ocr = None
        
        
        
        if self.use_minicpm:
            try:
                minicpm_config = minicpm_config or {}
                if(_can_load_model()):
                    self.minicpm_ocr = create_minicpm_ocr(**minicpm_config)
                else:
                    logger.warning("Insufficient storage to load MiniCPM-o 2.6 model. Falling back to traditional OCR.")
                    self.use_minicpm = False
                    self.minicpm_ocr = None
                
                logger.info("MiniCPM-o 2.6 initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize MiniCPM-o 2.6: {e}. Falling back to traditional OCR.")
                self.use_minicpm = False
                self.minicpm_ocr = None
    
    def _get_primary_ocr_engine(self) -> str:
        """Return the name of the primary OCR engine being used."""
        return "MiniCPM-o 2.6" if (self.use_minicpm and self.minicpm_ocr) else "FrameOCR + LLM"
    
    def _detect_video_format(self, video_path: Path) -> Dict[str, Any]:
        """
        Analyze video format to determine optimal processing strategy.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary with format information for optimization decisions
        """
        try:
            # Use ffprobe to get detailed format information
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            format_info = json.loads(result.stdout)
            
            # Extract key information for optimization
            video_stream = next((s for s in format_info['streams'] if s['codec_type'] == 'video'), {})
            audio_streams = [s for s in format_info['streams'] if s['codec_type'] == 'audio']
            
            analysis = {
                'video_codec': video_stream.get('codec_name', 'unknown'),
                'pixel_format': video_stream.get('pix_fmt', 'unknown'),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'has_audio': len(audio_streams) > 0,
                'audio_codec': audio_streams[0].get('codec_name', 'none') if audio_streams else 'none',
                'container_format': format_info['format'].get('format_name', 'unknown'),
                'duration': float(format_info['format'].get('duration', 0)),
                'size_bytes': int(format_info['format'].get('size', 0)),
                'can_stream_copy': self._can_use_stream_copy(video_stream, audio_streams)
            }
            
            logger.debug(f"Video format analysis: {analysis}")
            return analysis
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to analyze video format: {e}")
            return {
                'video_codec': 'unknown',
                'pixel_format': 'unknown', 
                'width': 1920,
                'height': 1080,
                'has_audio': True,
                'audio_codec': 'unknown',
                'container_format': 'unknown',
                'duration': 0,
                'size_bytes': 0,
                'can_stream_copy': False
            }
    
    def _can_use_stream_copy(self, video_stream: Dict, audio_streams: List[Dict]) -> bool:
        """
        Determine if we can use FFmpeg -c copy for fast processing.
        
        Args:
            video_stream: Video stream info from ffprobe
            audio_streams: Audio stream info from ffprobe
            
        Returns:
            True if stream copy is viable
        """
        # Common codecs that work well with stream copy
        good_video_codecs = {'h264', 'h265', 'hevc', 'vp8', 'vp9', 'av1'}
        good_audio_codecs = {'aac', 'mp3', 'opus', 'vorbis'}
        
        video_codec = video_stream.get('codec_name', '').lower()
        
        # Check video codec compatibility
        if video_codec not in good_video_codecs:
            logger.debug(f"Video codec {video_codec} not suitable for stream copy")
            return False
        
        # Check audio codec compatibility  
        for audio_stream in audio_streams:
            audio_codec = audio_stream.get('codec_name', '').lower()
            if audio_codec not in good_audio_codecs:
                logger.debug(f"Audio codec {audio_codec} not suitable for stream copy")
                return False
        
        # Check pixel format - some 10-bit formats need conversion
        pixel_format = video_stream.get('pix_fmt', '')
        if '10le' in pixel_format or '422' in pixel_format:
            logger.debug(f"Pixel format {pixel_format} may need conversion")
            return False
        
        return True
    
    def _create_named_pipe(self, suffix: str = ".mp4") -> Path:
        """
        Create a named pipe (FIFO) for streaming video data.
        
        Args:
            suffix: File extension for the pipe name
            
        Returns:
            Path to created named pipe
        """
        # Create temporary directory for pipes
        temp_dir = Path(tempfile.mkdtemp(prefix='video_pipes_'))
        pipe_path = temp_dir / f"stream{suffix}"
        
        try:
            # Create named pipe
            os.mkfifo(str(pipe_path))
            logger.debug(f"Created named pipe: {pipe_path}")
            return pipe_path
            
        except OSError as e:
            logger.error(f"Failed to create named pipe: {e}")
            # Fallback to regular temp file
            return temp_dir / f"fallback{suffix}"
    
    def _stream_copy_with_pixel_conversion(
        self, 
        input_video: Path, 
        output_video: Path,
        target_pixel_format: str = "yuv420p"
    ) -> bool:
        """
        Convert video with minimal re-encoding using pixel format conversion only.
        
        This is much faster than full re-encoding when only pixel format differs.
        
        Args:
            input_video: Source video path
            output_video: Destination video path  
            target_pixel_format: Target pixel format (default yuv420p for compatibility)
            
        Returns:
            True if conversion succeeded
        """
        try:
            # Build FFmpeg command for pixel format conversion only
            cmd = [
                'ffmpeg', '-i', str(input_video),
                '-vf', f'format={target_pixel_format}',  # Only convert pixel format
                '-c:v', 'libx264',  # Use efficient H.264 encoder
                '-preset', 'ultrafast',  # Fastest encoding preset
                '-crf', '18',  # High quality constant rate factor
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                '-y', str(output_video)
            ]
            
            logger.info(f"Converting pixel format: {input_video} -> {output_video}")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}");
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"Pixel conversion output: {result.stderr}")
            
            return output_video.exists() and output_video.stat().st_size > 0
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Pixel format conversion failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Pixel format conversion error: {e}")
            return False

    def _stream_copy_video(
        self,
        input_video: Path,
        output_video: Path, 
        format_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Copy video streams without re-encoding for maximum speed.
        
        Args:
            input_video: Source video path
            output_video: Destination video path
            format_info: Video format analysis (optional)
            
        Returns:
            True if stream copy succeeded
        """
        try:
            if format_info is None:
                format_info = self._detect_video_format(input_video)
            
            # Check if we can use pure stream copy
            if format_info['can_stream_copy']:
                cmd = [
                    'ffmpeg', '-i', str(input_video),
                    '-c', 'copy',  # Copy all streams without re-encoding
                    '-avoid_negative_ts', 'make_zero',
                    '-y', str(output_video)
                ]
                
                logger.info(f"Stream copying (no re-encoding): {input_video} -> {output_video}")
                
            else:
                # Need minimal conversion (usually just pixel format)
                pixel_fmt = format_info.get('pixel_format', 'unknown')
                
                if '10le' in pixel_fmt or '422' in pixel_fmt:
                    logger.info(f"Converting {pixel_fmt} to yuv420p for compatibility")
                    return self._stream_copy_with_pixel_conversion(input_video, output_video)
                else:
                    # Use fast presets for unknown formats
                    cmd = [
                        'ffmpeg', '-i', str(input_video),
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                        '-c:a', 'copy',
                        '-y', str(output_video)
                    ]
                    
                    logger.info(f"Fast re-encoding with stream copy audio: {input_video} -> {output_video}")
            
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"Stream copy output: {result.stderr}")
            
            return output_video.exists() and output_video.stat().st_size > 0
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Stream copy failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Stream copy error: {e}")
            return False

    def _mask_video_streaming(
        self, 
        input_video: Path, 
        mask_config: Dict[str, Any], 
        output_video: Path,
        use_named_pipe: bool = True
    ) -> bool:
        """
        Apply video masking using streaming approach with optional named pipes.
        
        Args:
            input_video: Path to input video file
            mask_config: Dictionary containing mask coordinates
            output_video: Path for output masked video
            use_named_pipe: Whether to use named pipes for streaming
            
        Returns:
            True if masking succeeded, False otherwise
        """
        try:
            format_info = self._detect_video_format(input_video)
            
            endoscope_x = mask_config.get("endoscope_image_x", 0)
            endoscope_y = mask_config.get("endoscope_image_y", 0)
            endoscope_w = mask_config.get("endoscope_image_width", 640)
            endoscope_h = mask_config.get("endoscope_image_height", 480)
            
            # Check if we can use simple crop (most efficient)
            if endoscope_y == 0 and endoscope_h == mask_config.get("image_height", 1080):
                # Simple crop - can often use stream copy for audio
                crop_filter = f"crop=in_w-{endoscope_x}:in_h:{endoscope_x}:0"
                
                if use_named_pipe and format_info['can_stream_copy']:
                    # Use named pipe for streaming processing
                    pipe_path = self._create_named_pipe(".mp4")
                    
                    try:
                        # Start background process to write to pipe
                        writer_cmd = [
                            'ffmpeg', '-i', str(input_video),
                            '-vf', crop_filter,
                            '-c:a', 'copy',  # Stream copy audio
                            '-f', 'mp4', '-movflags', 'faststart',
                            str(pipe_path)
                        ]
                        
                        # Start reader process from pipe
                        reader_cmd = [
                            'ffmpeg', '-i', str(pipe_path),
                            '-c', 'copy',  # Pure stream copy from pipe
                            '-y', str(output_video)
                        ]
                        
                        logger.info(f"Using named pipe for streaming mask: {crop_filter}")
                        
                        # Start writer in background
                        writer_proc = subprocess.Popen(
                            writer_cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE
                        )
                        
                        try:
                            # Start reader (blocks until complete)
                            try:
                                duration = format_info.get("duration", 0) or 1  # seconds
                                multiplier = 10.0                                # 10× realtime headroom (On server this might be unessecary)
                                reader_tmo  = max(600, duration * multiplier)   # at least 10 min
                                writer_tmo  = reader_tmo * 0.5                  # writer should finish sooner
                            except KeyError:
                                # Fallback to default timeouts if duration not available
                                reader_tmo = 1000000
                                logger.warning("FFProbe problem. Using default timeout for reader process")
                        
                            

                            reader_result = subprocess.run(
                                reader_cmd,
                                capture_output=True,
                                text=True,
                                check=True,
                                timeout=reader_tmo,
                            )

                            
                            # Wait for writer to complete with timeout
                            try:
                                writer_stdout, writer_stderr = writer_proc.communicate(timeout=writer_tmo)
                                writer_returncode = writer_proc.returncode
                                
                                # Check if both processes completed successfully
                                if writer_returncode != 0:
                                    logger.error(f"Writer process failed with code {writer_returncode}: {writer_stderr.decode() if isinstance(writer_stderr, bytes) else writer_stderr}")
                                    raise subprocess.CalledProcessError(writer_returncode, writer_cmd, writer_stderr)
                                
                                logger.debug(f"Streaming mask completed via named pipe")
                                
                            except subprocess.TimeoutExpired:
                                logger.error("Writer process timed out, terminating...")
                                writer_proc.terminate()
                                try:
                                    writer_proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    logger.error("Writer process did not terminate gracefully, killing...")
                                    writer_proc.kill()
                                    writer_proc.wait()
                                raise RuntimeError("Named pipe writer process timed out")
                                
                        except subprocess.TimeoutExpired:
                            logger.error("Reader process timed out, cleaning up...")
                            # Kill writer if reader times out
                            if writer_proc.poll() is None:  # Still running
                                writer_proc.terminate()
                                try:
                                    writer_proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    writer_proc.kill()
                                    writer_proc.wait()
                            raise RuntimeError("Named pipe reader process timed out")
                            
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Reader process failed: {e.stderr}")
                            # Clean up writer process
                            if writer_proc.poll() is None:  # Still running
                                writer_proc.terminate()
                                try:
                                    writer_proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    writer_proc.kill()
                                    writer_proc.wait()
                            raise
                            
                    finally:
                        # Ensure writer process is cleaned up
                        if writer_proc.poll() is None:  # Still running
                            logger.warning("Writer process still running during cleanup, terminating...")
                            writer_proc.terminate()
                            try:
                                writer_proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                logger.error("Writer process did not terminate, killing...")
                                writer_proc.kill()
                                writer_proc.wait()
                        
                        # Clean up pipe
                        if pipe_path.exists():
                            try:
                                pipe_path.unlink()
                                pipe_path.parent.rmdir()
                            except OSError:
                                pass
                else:
                    # Direct processing without pipe
                    cmd = [
                        'ffmpeg', '-i', str(input_video),
                        '-vf', crop_filter,
                        '-c:a', 'copy',  # Stream copy audio when possible
                        '-y', str(output_video)
                    ]
                    
                    logger.info(f"Direct crop masking: {crop_filter}")
                    logger.debug(f"FFmpeg command: {' '.join(cmd)}")
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    logger.debug(f"Direct masking output: {result.stderr}")
            
            else:
                # Complex masking - use drawbox filters
                mask_filters = []
                
                # Build mask rectangles (same logic as before)
                if endoscope_x > 0:
                    mask_filters.append(f"drawbox=0:0:{endoscope_x}:{mask_config.get('image_height', 1080)}:color=black@1:t=fill")
                
                right_start = endoscope_x + endoscope_w
                image_width = mask_config.get('image_width', 1920)
                if right_start < image_width:
                    right_width = image_width - right_start
                    mask_filters.append(f"drawbox={right_start}:0:{right_width}:{mask_config.get('image_height', 1080)}:color=black@1:t=fill")
                
                if endoscope_y > 0:
                    mask_filters.append(f"drawbox={endoscope_x}:0:{endoscope_w}:{endoscope_y}:color=black@1:t=fill")
                
                bottom_start = endoscope_y + endoscope_h
                image_height = mask_config.get('image_height', 1080)
                if bottom_start < image_height:
                    bottom_height = image_height - bottom_start
                    mask_filters.append(f"drawbox={endoscope_x}:{bottom_start}:{endoscope_w}:{bottom_height}:color=black@1:t=fill")
                
                vf = ','.join(mask_filters)
                
                # Use optimized encoding for complex masks
                cmd = [
                    'ffmpeg', '-i', str(input_video),
                    '-vf', vf,
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',  # Fast encoding
                    '-c:a', 'copy',  # Always copy audio
                    '-y', str(output_video)
                ]
                
                logger.info(f"Complex mask processing with {len(mask_filters)} regions")
                logger.debug(f"FFmpeg command: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.debug(f"Complex masking output: {result.stderr}")
            
            # Verify output
            if output_video.exists() and output_video.stat().st_size > 0:
                # Compare file sizes to ensure reasonable output
                input_size = input_video.stat().st_size
                output_size = output_video.stat().st_size
                size_ratio = output_size / input_size if input_size > 0 else 0
                
                if size_ratio < 0.1:  # Output suspiciously small
                    logger.warning(f"Output video much smaller than input ({size_ratio:.1%})")
                
                logger.info(f"Successfully created masked video: {output_video} (size ratio: {size_ratio:.1%})")
                return True
            else:
                logger.error("Masked video is empty or missing")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Streaming mask failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Streaming mask error: {e}")
            return False

    def remove_frames_from_video_streaming(
        self,
        original_video: Path, 
        frames_to_remove: List[int], 
        output_video: Path,
        total_frames: Optional[int] = None,
        use_named_pipe: bool = True
    ) -> bool:
        """
        Remove frames using streaming approach with optional named pipes.
        
        Args:
            original_video: Path to original video
            frames_to_remove: List of frame numbers to remove (0-based)
            output_video: Path for output video
            total_frames: Total frame count (for optimization)
            use_named_pipe: Whether to use named pipes for streaming
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not frames_to_remove:
                logger.info("No frames to remove, using stream copy")
                return self._stream_copy_video(original_video, output_video)
            
            format_info = self._detect_video_format(original_video)
            
            logger.info(f"Removing {len(frames_to_remove)} frames using streaming method")
            
            # Create frame selection filter
            idx_list = '+'.join([f'eq(n\\,{idx})' for idx in frames_to_remove])
            vf = f"select='not({idx_list})',setpts=N/FRAME_RATE/TB"
            af = f"aselect='not({idx_list})',asetpts=N/SR/TB"
            
            if use_named_pipe and len(frames_to_remove) < (total_frames or 1000) * 0.1:
                # Use named pipe for small frame removal operations
                pipe_path = self._create_named_pipe(".mp4")
                
                try:
                    # Pipeline: filter frames -> pipe -> stream copy to final output
                    filter_cmd = [
                        'ffmpeg', '-i', str(original_video),
                        '-vf', vf,
                        '-af', af, 
                        '-f', 'mp4', '-movflags', 'faststart',
                        str(pipe_path)
                    ]
                    
                    copy_cmd = [
                        'ffmpeg', '-i', str(pipe_path),
                        '-c', 'copy',  # Stream copy from pipe
                        '-y', str(output_video)
                    ]
                    
                    logger.info("Using named pipe for frame removal streaming")
                    
                    # Start filter process in background
                    filter_proc = subprocess.Popen(
                        filter_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Start copy process (blocks until complete)
                    copy_result = subprocess.run(
                        copy_cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    
                    # Wait for filter to complete
                    filter_result = filter_proc.communicate()
                    
                    logger.debug("Streaming frame removal completed via named pipe")
                    
                finally:
                    # Clean up pipe
                    if pipe_path.exists():
                        try:
                            pipe_path.unlink()
                            pipe_path.parent.rmdir()
                        except OSError:
                            pass
            
            else:
                # Direct processing for larger removals or when pipes unavailable
                if format_info['can_stream_copy'] and format_info['has_audio']:
                    # Use optimized encoding to preserve quality
                    cmd = [
                        'ffmpeg', '-i', str(original_video),
                        '-vf', vf,
                        '-af', af,
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                        '-c:a', 'aac', '-b:a', '128k',  # Re-encode audio with high quality
                        '-y', str(output_video)
                    ]
                else:
                    # Video-only or format needs re-encoding
                    cmd = [
                        'ffmpeg', '-i', str(original_video),
                        '-vf', vf,
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                        '-an' if not format_info['has_audio'] else '-af', 
                        af if format_info['has_audio'] else '',
                        '-y', str(output_video)
                    ]
                    
                    # Remove empty arguments
                    cmd = [arg for arg in cmd if arg]
                
                logger.info("Direct frame removal processing")
                logger.debug(f"FFmpeg command: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.debug(f"Direct frame removal output: {result.stderr}")
            
            # Verify output
            if output_video.exists() and output_video.stat().st_size > 0:
                logger.info(f"Successfully removed frames: {output_video}")
                return True
            else:
                logger.error("Frame removal output is empty or missing")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Streaming frame removal failed: {e.stderr}")
            
            # Fallback to original method without audio processing
            try:
                logger.warning("Retrying frame removal without audio processing...")
                cmd_no_audio = [
                    'ffmpeg', '-i', str(original_video),
                    '-vf', vf,
                    '-an',  # No audio
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-y', str(output_video)
                ]
                result = subprocess.run(cmd_no_audio, capture_output=True, text=True, check=True)
                logger.info("Successfully removed frames without audio")
                return True
            except subprocess.CalledProcessError as e2:
                logger.error(f"Frame removal fallback also failed: {e2.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Streaming frame removal error: {e}")
            return False

    def _process_frame(
        self,
        gray_frame: np.ndarray,
        endoscope_roi: dict | None,
    ) -> tuple[bool, dict, str, float]:
        """
        Centralised OCR + metadata extraction for ONE frame.

        Returns:
            is_sensitive, frame_metadata, ocr_text, ocr_conf
        """
        if self.use_minicpm and self.minicpm_ocr:
            pil_img = (
                Image.fromarray(gray_frame, mode="L")
                .convert("RGB")
            )
            try:
                is_sensitive, frame_metadata, ocr_text = (
                    # using minicpm
                    self.minicpm_ocr.detect_sensitivity_unified(
                        pil_img,
                        context="endoscopy video frame",
                    )
                )
            except ValueError as ve:
                logger.error(f"MiniCPM-o 2.6 processing failed: {ve}")
                self.use_minicpm = False
                self.minicpm_ocr = False
                logger.warning(
                    "MiniCPM-o 2.6 failed to detect sensitivity or text, falling back to traditional OCR."
                )
                # Fallback to traditional OCR
                ocr_text, ocr_conf, _ = self.frame_ocr.extract_text_from_frame(
                    gray_frame,
                    roi=endoscope_roi,
                    high_quality=True,
                )
                frame_metadata = (
                    self.PatientDataExtractor
                )
                is_sensitive = self.frame_metadata_extractor.is_sensitive_content(
                    frame_metadata
                )
            except Exception as e:
                logger.error(f"MiniCPM-o 2.6 processing failed: {e}")
                self.use_minicpm = False
                self.minicpm_ocr = False
                logger.warning(
                    "MiniCPM-o 2.6 failed to detect sensitivity or text, falling back to traditional OCR."
                )
                # Fallback to traditional OCR
                ocr_text, ocr_conf, _ = self.frame_ocr.extract_text_from_frame
            # MiniCPM does not provide a confidence value – treat as 1.0
            ocr_conf = 1.0
        else:
            ocr_text, ocr_conf, _ = self.frame_ocr.extract_text_from_frame(
                gray_frame,
                roi=endoscope_roi,
                high_quality=True,
            )
            frame_metadata = (
                self.frame_metadata_extractor.extract_metadata_from_frame_text(
                    ocr_text
                )
            )
            is_sensitive = self.frame_metadata_extractor.is_sensitive_content(
                frame_metadata
            )

        return is_sensitive, frame_metadata, ocr_text, ocr_conf

    # ------------------------------------------------------------------ main
    def clean_video(
        self,
        video_path: Path,
        video_file_obj=None,
        tmp_dir: Optional[Path] = None,
        device_name: Optional[str] = None,
        endoscope_roi: Optional[Dict[str, Any]] = None,
        processor_rois: Optional[Dict[str, Dict[str, Any]]] = None,
        output_path: Optional[Path] = None,
    ) -> tuple[Path, Dict[str, Any]]:
        """
        Refactored version: single code path, fewer duplicated branches.
        """

        if tmp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="frame_cleaner_"))

        if device_name is None:
            device_name = "olympus_cv_1500"

        output_video = (
            output_path or video_path.with_stem(f"{video_path.stem}_anony")
        )

        # -------- initialise extracted‑metadata accumulator -----------------
        accumulated: dict[str, Any] = {
            "patient_first_name": None,
            "patient_last_name": None,
            "patient_dob": None,
            "casenumber": None,
            "patient_gender": None,
            "examination_date": None,
            "examination_time": None,
            "examiner": None,
            "representative_ocr_text": None,
            "source": "frame_extraction",
        }

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ------ choose sampling parameters ----------------------------------
        long_video = total_frames > 10_000
        max_samples = (
            min(500, total_frames // 20) if long_video else total_frames
        )

        logger.info(
            "%s video detected (%d frames). Sampling ≤%d frames.",
            "Long" if long_video else "Short",
            total_frames,
            max_samples,
        )

        # --------------------------------------------------------------------
        sensitive_idx: list[int] = []
        sampled = 0

        for idx, gray_frame, stride in self._iter_video(
            video_path, total_frames
        ):
            if sampled >= max_samples:
                break
            sampled += 1

            is_sensitive, frame_meta, ocr_text, ocr_conf = self._process_frame(
                gray_frame, endoscope_roi
            )

            # merge metadata for every frame (high recall)
            accumulated = self.frame_metadata_extractor.merge_metadata(
                accumulated, frame_meta or {}
            )

            # collect text for preview (loose gates handled inside)
            if ocr_text:
                self.best_frame_text.push(
                    ocr_text, ocr_conf, is_sensitive=is_sensitive
                )

            if is_sensitive:
                sensitive_idx.append(idx)

        # representative preview text
        accumulated["representative_ocr_text"] = (
            self.best_frame_text.reduce().get("best", "")
        )

        sensitive_ratio = (
            len(sensitive_idx) / total_frames if total_frames else 0.0
        )
        logger.info(
            "Sensitive frames: %d/%d (%.1f %%)",
            len(sensitive_idx),
            total_frames,
            100 * sensitive_ratio,
        )

        # ------------- decide between frame removal and masking -------------
        try:
            if sensitive_ratio <= 0.10:
                # ---- low ratio ➞ remove individual frames ------------------
                logger.info("Using frame‑removal strategy.")
                ok = self.remove_frames_from_video_streaming(
                    video_path,
                    sensitive_idx,
                    output_video,
                    total_frames=total_frames,
                )
            else:
                # ---- high ratio ➞ mask overlay area -------------------------
                logger.info("Using masking strategy.")
                if endoscope_roi and self._validate_roi(endoscope_roi):
                    mask_cfg = self._create_mask_config_from_roi(
                        endoscope_roi, processor_rois
                    )
                else:
                    mask_cfg = self._load_mask(device_name)

                ok = self._mask_video_streaming(
                    video_path, mask_cfg, output_video, use_named_pipe=True
                )

            if not ok:
                logger.error("Processing failed – copying original video.")
                shutil.copy2(video_path, output_video)

        except Exception:
            logger.exception("Processing failed – copying original video.")
            shutil.copy2(video_path, output_video)
        
        finally:
            # Clean up temporary files
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # ----------------------- persist metadata ---------------------------
        if video_file_obj:
            self._update_video_sensitive_meta(video_file_obj, accumulated)

        return output_video, accumulated

            


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
            
            return False, None
            
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
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-y', str(output_video)
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
            
    def _iter_video(self, video_path: Path, total_frames: int) -> Iterator[Tuple[int, np.ndarray, int]]:
        """
        Yield (abs_frame_index, gray_frame, skip_value) with adaptive subsampling
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open %s", video_path)
            return
        

        skip = math.ceil(total_frames / 200)
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
            metadata = self.extract_metadata_deepseek(metadata.get('representative_ocr_text', ''))
            
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

    def _iter_video_adaptive(
        self,
        video_path: Path,
        total_frames: int,
        max_samples: int = 500,
        stop_tolerance: float = 0.10,
        conf_z: float = 1.96,
        min_samples: int = 30
    ) -> Iterator[Tuple[int, np.ndarray, int, bool]]:
        """
        Adaptive video sampling with statistical early stopping.
        
        Yields (abs_idx, gray_frame, stride, should_continue) until we are confident
        that the sensitive-ratio is either > stop_tolerance or <= stop_tolerance.
        
        Args:
            video_path: Path to video file
            total_frames: Total number of frames in video
            max_samples: Maximum frames to sample (default 500)
            stop_tolerance: Decision boundary for sensitive ratio (default 0.10)
            conf_z: Confidence interval Z-score (default 1.96 for 95%)
            min_samples: Minimum samples before testing confidence (default 30)
            
        Yields:
            Tuple of (frame_index, gray_frame, stride, should_continue)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
        
        # Uniform stride ensuring ≤ max_samples if we run full pass
        stride = max(1, math.ceil(total_frames / max_samples))
        
        tested, hits = 0, 0
        frame_idx = -1
        
        try:
            while True:
                ok, bgr = cap.read()
                frame_idx += 1
                if not ok:
                    break  # End of stream
                    
                if frame_idx % stride != 0:
                    continue  # Skip non-sampled frame
                
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                tested += 1
                
                # Yield frame and receive sensitivity result back
                is_sensitive = yield frame_idx, gray, stride, True
                
                if is_sensitive:
                    hits += 1
                
                # Early-stop test (Wilson/Wald interval) after minimum samples
                if tested >= min_samples:
                    phat = hits / tested
                    half_w = conf_z * math.sqrt(phat * (1 - phat) / tested)
                    lo, hi = phat - half_w, phat + half_w
                    
                    if hi < stop_tolerance:
                        logger.info(f"Early stop: confident that ratio ≤ {stop_tolerance:.1%} "
                                   f"(tested={tested}, hits={hits}, CI=[{lo:.3f}, {hi:.3f}])")
                        # Yield final frame with should_continue=False
                        yield frame_idx, gray, stride, False
                        break
                        
                    if lo > stop_tolerance:
                        logger.info(f"Early stop: confident that ratio > {stop_tolerance:.1%} "
                                   f"(tested={tested}, hits={hits}, CI=[{lo:.3f}, {hi:.3f}])")
                        # Yield final frame with should_continue=False
                        yield frame_idx, gray, stride, False
                        break
                
                if tested >= max_samples:
                    logger.info(f"Reached sample budget: {max_samples} frames tested")
                    break
                    
        finally:
            cap.release()
    
    def _sample_frames_coroutine(
        self,
        video_path: Path,
        total_frames: int,
        max_samples: int = 500,
        stop_tolerance: float = 0.10
    ) -> Tuple[List[int], float, bool]:
        """
        Sample frames using statistical adaptive method and return sensitivity analysis.
        
        Args:
            video_path: Path to video file
            total_frames: Total number of frames
            max_samples: Maximum frames to sample
            stop_tolerance: Decision boundary (default 0.10)
            
        Returns:
            Tuple of (sensitive_frame_indices, estimated_ratio, early_stopped)
        """
        sensitive_frame_indices = []  # Fix: renamed for consistency
        tested_count = 0
        hits_count = 0
        early_stopped = False
        
        # Initialize the adaptive iterator
        frame_iter = self._iter_video_adaptive(
            video_path, total_frames, max_samples, stop_tolerance
        )
        
        try:
            # Get first frame
            frame_idx, gray_frame, stride, should_continue = next(frame_iter)
            
            while should_continue:
                tested_count += 1
                
                # Use MiniCPM-o unified detection if available, otherwise fallback
                try:
                    if self.use_minicpm and self.minicpm_ocr:
                        # Convert numpy array to PIL Image for MiniCPM-o
                        if len(gray_frame.shape) == 2:  # Grayscale
                            image = Image.fromarray(gray_frame, mode='L').convert('RGB')
                        else:  # Color
                            image = Image.fromarray(gray_frame, mode='RGB')
                        
                        is_sensitive, frame_metadata, ocr_text = self.minicpm_ocr.detect_sensitivity_unified(
                            image, context="endoscopy video frame"
                        )
                        
                        logger.debug(f"MiniCPM-o analysis for frame {frame_idx}: sensitive={is_sensitive}")
                        
                    else:
                        # Fallback to traditional OCR + LLM approach
                        ocr_text, avg_conf, _ = self.frame_ocr.extract_text_from_frame(
                            gray_frame, 
                            roi=None,  # Will be passed from caller if available
                            high_quality=True
                        )
                        
                        is_sensitive = False
                        frame_metadata = {}
                        
                        if ocr_text:
                            # Feed the 'best text' sampler
                            self.best_frame_text.push(ocr_text, avg_conf)
                            
                            # Try LLM extraction first (faster for long videos)
                            frame_metadata = self.extract_metadata_deepseek(ocr_text)
                            if not frame_metadata:
                                frame_metadata = self.frame_metadata_extractor.extract_metadata_from_frame_text(ocr_text)
                            
                            # Check if frame contains sensitive content
                            is_sensitive = self.frame_metadata_extractor.is_sensitive_content(frame_metadata)
                        
                        logger.debug(f"Traditional OCR analysis for frame {frame_idx}: sensitive={is_sensitive}")
                        
                except Exception as e:
                    logger.exception(f"Error during sensitivity detection for frame {frame_idx}: {e}")
                    is_sensitive = False  # Fail-safe: treat as non-sensitive on error
                
                if is_sensitive:
                    self.reservoir.append(ocr_text)        # no thresholds needed
                    sensitive_frame_indices.append(frame_idx)
                    hits_count += 1
                    logger.info(f"Found sensitive data in frame {frame_idx}")
                
                # Send result back to iterator and get next frame
                try:
                    frame_idx, gray_frame, stride, should_continue = frame_iter.send(is_sensitive)
                    if not should_continue:
                        early_stopped = True
                        break
                except StopIteration:
                    break
                    
        except StopIteration:
            # Iterator finished normally
            pass
        
        # Calculate final ratio
        estimated_ratio = hits_count / tested_count if tested_count > 0 else 0.0
        early_stopped = tested_count < max_samples
        
        logger.info(f"Statistical sampling complete: {hits_count}/{tested_count} = {estimated_ratio:.1%} "
                   f"(early_stopped={early_stopped})")
        
        return sensitive_frame_indices, estimated_ratio, early_stopped

    def analyze_video_sensitivity(self) -> Dict[str, Any]:
        """
        Analyze video for sensitive content without processing
        Returns analysis metadata
        """
        logger.info(f"Analyzing video sensitivity: {self.video_path}")
        
        try:
            # Get total frame count
            total_frames = self._get_total_frames()
            logger.info(f"Total frames in video: {total_frames}")
            
            # Detect sensitive frames using streaming
            sensitive_frames = list(self._detect_sensitive_frames_streaming(total_frames))
            logger.info(f"Detected {len(sensitive_frames)} sensitive frames")
            
            # Calculate sensitivity ratio
            sensitivity_ratio = len(sensitive_frames) / total_frames if total_frames > 0 else 0.0
            
            # Get video metadata
            video_info = self._get_video_info()
            
            analysis_result = {
                "sensitive_frames": len(sensitive_frames),
                "total_frames": total_frames,
                "sensitivity_ratio": sensitivity_ratio,
                "duration": video_info.get("duration"),
                "resolution": video_info.get("resolution"),
                "recommended_method": "masking" if sensitivity_ratio > 0.1 else "frame_removal",
                "sensitive_frame_list": sensitive_frames[:100] if len(sensitive_frames) <= 100 else sensitive_frames[:100] + ["...truncated"],
                "analysis_engine": "minicpm" if self.use_minicpm else "traditional"
            }
            
            logger.info(f"Analysis complete: {analysis_result['recommended_method']} recommended")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            raise

    def _get_video_info(self) -> Dict[str, Any]:
        """Get basic video information using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
                str(self.video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            info = {}
            if video_stream:
                info['resolution'] = f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}"
                
            format_info = data.get('format', {})
            duration = format_info.get('duration')
            if duration:
                info['duration'] = float(duration)
                
            return info
            
        except Exception as e:
            logger.warning(f"Failed to get video info: {e}")
            return {}