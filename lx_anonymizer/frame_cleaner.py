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
# from tkinter import N
from typing import List, Optional, Tuple, Dict, Any, Union, Iterator
import cv2
import numpy as np
from PIL import Image

from lx_anonymizer.frame_ocr import FrameOCR
from lx_anonymizer.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.best_frame_text import BestFrameText
from lx_anonymizer.utils.ollama import ensure_ollama
from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.minicpm_ocr import MiniCPMVisionOCR, create_minicpm_ocr, _can_load_model
from lx_anonymizer.spacy_extractor import PatientDataExtractor
from lx_anonymizer.ocr import trocr_full_image_ocr
from typing_inspection.typing_objects import NoneType

from lx_anonymizer.ollama_llm_meta_extraction_optimized import (
    OllamaOptimizedExtractor, 
    EnrichedMetadataExtractor, 
    FrameSamplingOptimizer,
    create_optimized_extractor_with_sampling
)

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
    - NVIDIA NVENC hardware acceleration with CPU fallback
    """
    
    def __init__(self, use_minicpm: bool = True, minicpm_config: Optional[Dict[str, Any]] = None, use_llm: bool = False):
        # Initialize specialized frame processing components
        self.frame_ocr = FrameOCR()
        self.frame_metadata_extractor = FrameMetadataExtractor()
        self.PatientDataExtractor = PatientDataExtractor()
        
        # Enhanced OCR integration - use enhanced components if available
        logger.info("Initializing with Enhanced OCR components (OCR_FIX_V1 enabled)")
        self.best_frame_text = BestFrameText()
        self.use_enhanced_ocr = True
        
        # LLM usage flag (guard)
        self.use_llm = bool(use_llm)
        
        # Initialize MiniCPM-o 2.6 if enabled
        self.use_minicpm = use_minicpm
        self.minicpm_ocr = create_minicpm_ocr() if use_minicpm else None
        self._log_hf_cache_info()
        
        # Initialize the optimized ollama processing pipeline (guarded)
        self.ollama_proc = None
        self.ollama_extractor = None
        self.frame_sampling_optimizer = None
        self.enriched_extractor = None
        if self.use_llm:
            try:
                # Initialize Ollama for LLM processing
                self.ollama_proc = ensure_ollama()
                self.ollama_extractor = OllamaOptimizedExtractor()
                # Initialize enriched metadata extraction components
                self.frame_sampling_optimizer = FrameSamplingOptimizer(max_frames=100, skip_similar_threshold=0.85)
                self.enriched_extractor = EnrichedMetadataExtractor(
                    ollama_extractor=self.ollama_extractor,
                    frame_optimizer=self.frame_sampling_optimizer
                )
            except Exception as e:
                logger.warning(f"Ollama/LLM unavailable, disabling LLM features: {e}")
                self.use_llm = False
                self.ollama_proc = None
                self.ollama_extractor = None
                self.frame_sampling_optimizer = None
                self.enriched_extractor = None
        
        # Frame data collection for batch processing
        self.frame_collection = []
        self.ocr_text_collection = []
        self.current_video_total_frames = 0
        
        # Hardware acceleration detection
        self.nvenc_available = self._detect_nvenc_support()
        self.preferred_encoder = self._get_preferred_encoder()
        
        logger.info(f"Hardware acceleration: NVENC {'available' if self.nvenc_available else 'not available'}")
        logger.info(f"Using encoder: {self.preferred_encoder}")
        
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
    
    def _detect_nvenc_support(self) -> bool:
        """
        Detect if NVIDIA NVENC hardware acceleration is available.
        
        Returns:
            True if NVENC is available, False otherwise
        """
        try:
            # Test NVENC availability with a minimal command (minimum size for NVENC)
            cmd = [
                'ffmpeg', '-nostdin', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=256x256:rate=1',
                '-c:v', 'h264_nvenc', '-preset', 'p1', '-f', 'null', '-'
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=15,
                check=False
            )
            
            if result.returncode == 0:
                logger.debug("NVENC h264 encoding test successful")
                return True
            else:
                logger.debug(f"NVENC test failed: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"NVENC detection failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error during NVENC detection: {e}")
            return False
    
    def _get_preferred_encoder(self) -> Dict[str, Any]:
        """
        Get the preferred video encoder configuration based on available hardware.
        
        Returns:
            Dictionary with encoder configuration
        """
        if self.nvenc_available:
            return {
                'name': 'h264_nvenc',
                'preset_param': '-preset',
                'preset_value': 'p4',  # Medium quality/speed for NVENC (P1=fastest, P7=best quality)
                'quality_param': '-cq',
                'quality_value': '20',  # NVENC CQ mode (lower = better quality)
                'type': 'nvenc',
                'fallback_preset': 'p1'  # Fastest NVENC preset for fallback
            }
        else:
            return {
                'name': 'libx264',
                'preset_param': '-preset',
                'preset_value': 'veryfast',  # CPU preset
                'quality_param': '-crf',
                'quality_value': '18',  # CPU CRF mode
                'type': 'cpu',
                'fallback_preset': 'ultrafast'  # Fastest CPU preset for fallback
            }
    
    def _build_encoder_cmd(self, 
                          quality_mode: str = 'balanced',
                          fallback: bool = False) -> List[str]:
        """
        Build encoder command arguments based on available hardware and quality requirements.
        
        Args:
            quality_mode: 'fast', 'balanced', or 'quality'
            fallback: Whether to use fallback settings for compatibility
            
        Returns:
            List of FFmpeg encoder arguments
        """
        encoder = self.preferred_encoder
        
        if encoder['type'] == 'nvenc':
            # NVIDIA NVENC configuration
            if fallback:
                preset = encoder['fallback_preset']  # p1 - fastest
                quality = '28'  # Lower quality for speed
            elif quality_mode == 'fast':
                preset = 'p2'  # Faster preset
                quality = '25'
            elif quality_mode == 'quality':
                preset = 'p6'  # Higher quality preset
                quality = '18'
            else:  # balanced
                preset = encoder['preset_value']  # p4
                quality = encoder['quality_value']  # 20
            
            return [
                '-c:v', encoder['name'],
                encoder['preset_param'], preset,
                encoder['quality_param'], quality,
                '-gpu', '0',  # Use first GPU
                '-rc', 'vbr',  # Variable bitrate
                '-profile:v', 'high'
            ]
        else:
            # CPU libx264 configuration
            if fallback:
                preset = encoder['fallback_preset']  # ultrafast
                quality = '23'  # Lower quality for speed
            elif quality_mode == 'fast':
                preset = 'faster'
                quality = '20'
            elif quality_mode == 'quality':
                preset = 'slow'
                quality = '15'
            else:  # balanced
                preset = encoder['preset_value']  # veryfast
                quality = encoder['quality_value']  # 18
            
            return [
                '-c:v', encoder['name'],
                encoder['preset_param'], preset,
                encoder['quality_param'], quality,
                '-profile:v', 'high'
            ]
    
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
                'ffprobe', '-nostdin', '-v', 'quiet', '-print_format', 'json', 
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
            # Get optimal encoder configuration
            encoder_args = self._build_encoder_cmd('balanced')
            
            # Build FFmpeg command for pixel format conversion with hardware acceleration
            cmd = [
                'ffmpeg', '-nostdin', '-y', '-i', str(input_video),
                '-vf', f'format={target_pixel_format}',  # Only convert pixel format
                *encoder_args,  # Use hardware-optimized encoder
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                str(output_video)
            ]
            
            logger.info(f"Converting pixel format: {input_video} -> {output_video} (using {self.preferred_encoder['type']})")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"Pixel conversion output: {result.stderr}")
            
            return output_video.exists() and output_video.stat().st_size > 0
            
        except subprocess.CalledProcessError as e:
            # Try fallback encoder if hardware acceleration fails
            if self.preferred_encoder['type'] == 'nvenc':
                logger.warning(f"NVENC pixel conversion failed, trying CPU fallback: {e.stderr}")
                return self._pixel_conversion_fallback(input_video, output_video, target_pixel_format)
            else:
                logger.error(f"Pixel format conversion failed: {e.stderr}")
                return False
            logger.error(f"Pixel format conversion failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Pixel format conversion error: {e}")
            return False

    def _pixel_conversion_fallback(self, 
                                  input_video: Path, 
                                  output_video: Path, 
                                  target_pixel_format: str) -> bool:
        """
        Fallback pixel format conversion using CPU encoding.
        
        Args:
            input_video: Source video path
            output_video: Destination video path
            target_pixel_format: Target pixel format
            
        Returns:
            True if conversion succeeded
        """
        try:
            cmd = [
                'ffmpeg', '-nostdin', '-y', '-i', str(input_video),
                '-vf', f'format={target_pixel_format}',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
                '-c:a', 'copy',
                '-avoid_negative_ts', 'make_zero',
                str(output_video)
            ]
            
            logger.info(f"CPU fallback pixel conversion: {input_video} -> {output_video}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return output_video.exists() and output_video.stat().st_size > 0
            
        except subprocess.CalledProcessError as e:
            logger.error(f"CPU fallback pixel conversion failed: {e.stderr}")
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
                    'ffmpeg', '-nostdin', '-y', '-i', str(input_video),
                    '-c', 'copy',  # Copy all streams without re-encoding
                    '-avoid_negative_ts', 'make_zero',
                    str(output_video)
                ]
                
                logger.info(f"Stream copying (no re-encoding): {input_video} -> {output_video}")
                
            else:
                # Need minimal conversion (usually just pixel format)
                pixel_fmt = format_info.get('pixel_format', 'unknown')
                
                if '10le' in pixel_fmt or '422' in pixel_fmt:
                    logger.info(f"Converting {pixel_fmt} to yuv420p for compatibility")
                    return self._stream_copy_with_pixel_conversion(input_video, output_video)
                else:
                    # Use hardware-optimized encoding for unknown formats
                    encoder_args = self._build_encoder_cmd('fast')
                    cmd = [
                        'ffmpeg', '-nostdin', '-y', '-i', str(input_video),
                        *encoder_args,
                        '-c:a', 'copy',
                        str(output_video)
                    ]
                    
                    logger.info(f"Fast re-encoding with {self.preferred_encoder['type']} and stream copy audio: {input_video} -> {output_video}")
            
            logger.debug(f"FFmpeg command with -nostdin: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"Stream copy output: {result.stderr}")
            
            return output_video.exists() and output_video.stat().st_size > 0
            
        except subprocess.CalledProcessError as e:
            # Try fallback if hardware acceleration fails
            if self.preferred_encoder['type'] == 'nvenc' and not format_info.get('can_stream_copy', False):
                logger.warning(f"NVENC stream copy failed, trying CPU fallback: {e.stderr}")
                return self._stream_copy_fallback(input_video, output_video, format_info)
            else:
                logger.error(f"Stream copy failed: {e.stderr}")
                return False
        except Exception as e:
            logger.error(f"Stream copy error: {e}")
            return False

    def _stream_copy_fallback(self, 
                             input_video: Path, 
                             output_video: Path, 
                             format_info: Dict[str, Any]) -> bool:
        """
        Fallback stream copy using CPU encoding.
        
        Args:
            input_video: Source video path
            output_video: Destination video path
            format_info: Video format information
            
        Returns:
            True if conversion succeeded
        """
        try:
            cmd = [
                'ffmpeg', '-nostdin', '-y', '-i', str(input_video),
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
                '-c:a', 'copy',
                str(output_video)
            ]
            
            logger.info(f"CPU fallback stream copy: {input_video} -> {output_video}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return output_video.exists() and output_video.stat().st_size > 0
            
        except subprocess.CalledProcessError as e:
            logger.error(f"CPU fallback stream copy failed: {e.stderr}")
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
            endoscope_x = mask_config.get("endoscope_image_x", 0)
            endoscope_y = mask_config.get("endoscope_image_y", 0)
            endoscope_w = mask_config.get("endoscope_image_width", 640)
            endoscope_h = mask_config.get("endoscope_image_height", 480)
            
            
            
            # Check if we can use simple crop (most efficient)
            if endoscope_y == 0 and endoscope_h == mask_config.get("image_height", 1080):
                # Simple crop - use single-pass processing for maximum efficiency
                crop_filter = f"crop=in_w-{endoscope_x}:in_h:{endoscope_x}:0"
                encoder_args = self._build_encoder_cmd('balanced')
                
                cmd = [
                    'ffmpeg', '-nostdin', '-y', '-i', str(input_video),
                    '-vf', crop_filter,
                    *encoder_args,
                    '-c:a', 'copy',
                    '-movflags', '+faststart',
                    str(output_video)
                ]
                
                logger.info(f"Direct crop masking (single pass) using {self.preferred_encoder['type']}: {crop_filter}")
                logger.debug(f"FFmpeg command with -nostdin: {' '.join(cmd)}")
                
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
                
                # Use hardware-optimized encoding for complex masks
                encoder_args = self._build_encoder_cmd('fast')
                cmd = [
                    'ffmpeg', '-nostdin', '-y', '-i', str(input_video),
                    '-vf', vf,
                    *encoder_args,  # Use hardware-optimized encoder
                    '-c:a', 'copy',  # Always copy audio
                    '-movflags', '+faststart',
                    str(output_video)
                ]
                
                logger.info(f"Complex mask processing with {len(mask_filters)} regions using {self.preferred_encoder['type']}")
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
                        'ffmpeg', '-nostdin', '-y', '-i', str(original_video),
                        '-vf', vf,
                        '-af', af, 
                        '-f', 'matroska',  # Use MKV for better streaming compatibility
                        str(pipe_path)
                    ]
                    
                    copy_cmd = [
                        'ffmpeg', '-nostdin', '-y', '-fflags', 'nobuffer', '-i', str(pipe_path),
                        '-c', 'copy',  # Stream copy from pipe
                        '-movflags', '+faststart',
                        str(output_video)
                    ]
                    
                    logger.info("Using named pipe for frame removal streaming (MKV container)")
                    logger.debug(f"Filter command with -nostdin: {' '.join(filter_cmd)}")
                    logger.debug(f"Copy command with -nostdin: {' '.join(copy_cmd)}")
                    
                    # Start filter process in background
                    filter_proc = subprocess.Popen(
                        filter_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Start copy process (blocks until complete)
                    subprocess.run(
                        copy_cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    
                    # Wait for filter to complete
                    filter_proc.communicate()
                    
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
                    # Use hardware-optimized encoding to preserve quality
                    encoder_args = self._build_encoder_cmd('balanced')
                    cmd = [
                        'ffmpeg', '-nostdin', '-y', '-i', str(original_video),
                        '-vf', vf,
                        '-af', af,
                        *encoder_args,  # Use hardware-optimized encoder
                        '-c:a', 'aac', '-b:a', '128k',  # Re-encode audio with high quality
                        '-movflags', '+faststart',
                        str(output_video)
                    ]
                else:
                    # Video-only or format needs re-encoding
                    encoder_args = self._build_encoder_cmd('balanced')
                    cmd = [
                        'ffmpeg', '-nostdin', '-y', '-i', str(original_video),
                        '-vf', vf,
                        *encoder_args,  # Use hardware-optimized encoder
                        '-an' if not format_info['has_audio'] else '-af', 
                        af if format_info['has_audio'] else '',
                        '-movflags', '+faststart' if format_info['has_audio'] else '',
                        str(output_video)
                    ]
                    
                    # Remove empty arguments
                    cmd = [arg for arg in cmd if arg]
                
                logger.info(f"Direct frame removal processing using {self.preferred_encoder['type']}")
                logger.debug(f"FFmpeg command with -nostdin: {' '.join(cmd)}")
                
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
            
            # Fallback to CPU method without audio processing
            try:
                logger.warning("Retrying frame removal without audio processing using CPU...")
                fallback_encoder_args = self._build_encoder_cmd('fast', fallback=True)
                cmd_no_audio = [
                    'ffmpeg', '-nostdin', '-y', '-i', str(original_video),
                    '-vf', vf,
                    '-an',  # No audio
                    *fallback_encoder_args,
                    str(output_video)
                ]
                result = subprocess.run(cmd_no_audio, capture_output=True, text=True, check=True)
                logger.info("Successfully removed frames without audio using CPU fallback")
                return True
            except subprocess.CalledProcessError as e2:
                logger.error(f"Frame removal CPU fallback also failed: {e2.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Streaming frame removal error: {e}")
            return False

    def _unified_metadata_extract(self, text: str) -> Dict[str, Any]:
        """Hierarchische Metadaten-Extraktion: LLM → spaCy → Regex-Fallback."""
        meta = {}
        if self.use_llm and self.ollama_extractor:
            try:
                meta = self.ollama_extractor.extract_metadata(text).model_dump()
            except Exception:
                meta = {}
        if not meta and hasattr(self.PatientDataExtractor, "extract"):
            try:
                meta = self.PatientDataExtractor.extract(text)
            except Exception:
                meta = {}
        if not meta:
            meta = self.frame_metadata_extractor.extract_metadata_from_frame_text(text)
        return meta

    def _process_frame_single(
        self,
        gray_frame: np.ndarray,
        endoscope_roi: dict | None = None,
        frame_id: int | None = None,
        extended: bool = False,
        collect_for_batch: bool = False,
    ) -> tuple[bool, dict, str, float]:
        """
        Konsolidierte Einzel-Frame-Verarbeitung mit OCR, Metadaten und optionaler Batch-Sammlung.
        """
        logger.debug(f"Processing frame_id={frame_id or 'unknown'}")
        ocr_text, ocr_conf, frame_metadata, is_sensitive = None, 0.0, {}, False

        if self.use_minicpm and self.minicpm_ocr:
            ocr_text, ocr_conf, frame_metadata, is_sensitive = self._ocr_with_minicpm(gray_frame)
        else:
            ocr_text, ocr_conf, frame_metadata, is_sensitive = self._ocr_with_tesserocr(gray_frame, endoscope_roi)

        # Einheitliche Metadaten-Extraktion
        if ocr_text:
            meta_unified = self._unified_metadata_extract(ocr_text)
            frame_metadata = self.frame_metadata_extractor.merge_metadata(frame_metadata, meta_unified)

        # Optional: Für Batch-Enrichment sammeln
        if collect_for_batch and ocr_text:
            self.frame_collection.append({
                "frame_id": frame_id,
                "ocr_text": ocr_text,
                "meta": frame_metadata,
                "is_sensitive": is_sensitive,
            })

        # Optional: BestFrameText für Preview
        if hasattr(self.best_frame_text, "push"):
            self.best_frame_text.push(ocr_text, ocr_conf)

        return is_sensitive, frame_metadata, ocr_text, ocr_conf

    def process_frames(
        self,
        frames: list[np.ndarray],
        endoscope_roi: dict | None = None,
        extended: bool = False,
    ) -> list[tuple[bool, dict, str, float]]:
        """
        Batch-Verarbeitung aller Frames mit optionalem Batch-Enrichment.
        """
        results = []
        self.frame_collection = []  # Reset für neuen Batch
        for i, gray_frame in enumerate(frames):
            logger.debug(f"Processing batch frame {i + 1}/{len(frames)}")
            is_sensitive, meta, text, conf = self._process_frame_single(
                gray_frame, endoscope_roi=endoscope_roi, frame_id=i, extended=extended, collect_for_batch=extended
            )
            results.append((is_sensitive, meta, text, conf))
        # Nach Batch: Metadaten-Anreicherung
        if extended and self.frame_collection:
            batch_meta = self._extract_enriched_metadata_batch()
            if batch_meta:
                logger.info(f"Merging batch-enriched metadata with accumulated results")
                for idx, (is_sensitive, meta, text, conf) in enumerate(results):
                    results[idx] = (is_sensitive, self.frame_metadata_extractor.merge_metadata(meta, batch_meta), text, conf)
        return results

    def clean_video(
        self,
        video_path: Path,
        video_file_obj=None,
        tmp_dir: Optional[Path] = None,
        device_name: Optional[str] = None,
        endoscope_roi: Optional[Dict[str, Any]] = None,
        processor_rois: Optional[Dict[str, Dict[str, Any]]] = None,
        output_path: Optional[Path] = None,
        technique: str = "mask_overlay",
        extended: bool = False,
    ) -> tuple[Path, Dict[str, Any]]:
        """
        Refactored version: single code path, fewer duplicated branches. Jetzt mit Batch-Metadaten-Logik.
        """
        if tmp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="frame_cleaner_"))
        if device_name is None:
            device_name = "olympus_cv_1500"
        output_video = (
            output_path or video_path.with_stem(f"{video_path.stem}_anony")
        )
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
        sensitive_idx: list[int] = []
        sampled = 0
        self.frame_collection = []  # Reset für neuen Batch
        for idx, gray_frame, stride in self._iter_video(
            video_path, total_frames
        ):
            if sampled >= max_samples:
                break
            sampled += 1
            is_sensitive, frame_meta, ocr_text, ocr_conf = self._process_frame_single(
                gray_frame, endoscope_roi, frame_id=idx, extended=extended, collect_for_batch=extended
            )
            accumulated = self.frame_metadata_extractor.merge_metadata(
                accumulated, frame_meta
            )
            if is_sensitive:
                sensitive_idx.append(idx)
        # Batch-Metadaten-Anreicherung nach Frame-Loop
        if extended and self.frame_collection:
            batch_enriched = self._extract_enriched_metadata_batch()
            if batch_enriched:
                accumulated = self.frame_metadata_extractor.merge_metadata(accumulated, batch_enriched)
        sensitive_ratio = (
            len(sensitive_idx) / total_frames if total_frames else 0.0
        )
        logger.info(
            "Sensitive frames: %d/%d (%.1f %%)",
            len(sensitive_idx),
            total_frames,
            100 * sensitive_ratio,
        )
        try:
            if technique == "remove_frames":
                logger.info("Using frame‑removal strategy.")
                ok = self.remove_frames_from_video_streaming(
                    video_path,
                    sensitive_idx,
                    output_video,
                    total_frames=total_frames,
                )
            elif technique == "mask_overlay":
                logger.info("Using masking strategy.")
                if endoscope_roi and self._validate_roi(endoscope_roi):
                    mask_cfg = endoscope_roi
                else:
                    mask_cfg = {"image_width": 1920, "image_height": 1080}
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
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        if video_file_obj:
            self._update_video_sensitive_meta(video_file_obj, accumulated)
        return output_video, accumulated
            


    def extract_frames(self, video_path: Path, output_dir: Path, max_frames: Optional[int] = None) -> List[Path]:
        """
        Extract frames from video using ffmpeg with high quality settings optimized for OCR.
        
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
        
        # Build high-quality ffmpeg command for OCR-optimized frame extraction
        # Use PNG for lossless extraction to preserve text quality
        cmd = [
            'ffmpeg', '-nostdin', '-y', '-i', str(video_path),
            '-vf', 'fps=1',  # Extract 1 frame per second (adjust as needed)
            '-q:v', '1',  # Highest quality (1-31, lower is better)
            '-pix_fmt', 'rgb24',  # RGB colorspace for maximum quality
            str(output_dir / 'frame_%04d.png')  # PNG for lossless compression
        ]
        
        # Limit frames if specified
        if max_frames:
            cmd.insert(-1, '-frames:v')
            cmd.insert(-1, str(max_frames))
        
        try:
            logger.info(f"Extracting high-quality frames from {video_path} to {output_dir}")
            logger.debug(f"FFmpeg command with -nostdin: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"ffmpeg output: {result.stderr}")
            
            # Get list of created frame files (now PNG)
            frame_files = sorted(output_dir.glob('frame_*.png'))
            logger.info(f"Extracted {len(frame_files)} high-quality frames")
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
            # Load image and convert to numpy array for FrameOCR
            image = Image.open(frame_path)
            
            # Convert to grayscale numpy array
            if image.mode != 'L':
                image = image.convert('L')
            
            frame_array = np.array(image)
            
            # Use FrameOCR with preprocessing instead of raw pytesseract
            ocr_text, ocr_conf, _ = self.frame_ocr.extract_text_from_frame(
                frame_array,
                roi=None,  # Could be enhanced with ROI if available
                high_quality=True
            )
            
            logger.debug(f"OCR confidence for {frame_path.name}: {ocr_conf:.3f} OCR Text:{ocr_text}")
            
            if not ocr_text.strip():
                logger.debug(f"No text detected in frame {frame_path.name}")
                return False
            
            logger.debug(f"OCR text from {frame_path.name} (conf={ocr_conf:.3f}): {ocr_text[:100]}...")
            
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
            # Load image and convert to numpy array for FrameOCR
            image = Image.open(frame_path)
            
            # Convert to grayscale numpy array
            if image.mode != 'L':
                image = image.convert('L')
            
            frame_array = np.array(image)
            
            # Use FrameOCR with preprocessing instead of raw pytesseract
            ocr_text, ocr_conf, _ = self.frame_ocr.extract_text_from_frame(
                frame_array,
                roi=None,  # Could be enhanced with ROI if available
                high_quality=True
            )
            logger.debug(f"OCR confidence: {ocr_conf:.3f} OCR Text: {ocr_text[:100]}...")

            if not ocr_text.strip():
                logger.debug(f"No text detected in frame {frame_path.name}")
                return False, None
            
            logger.debug(f"OCR text from {frame_path.name} (conf={ocr_conf:.3f}): {ocr_text[:100]}...")
            
            # Use LLM-powered metadata extraction
            try:
                frame_metadata = self.frame_metadata_extractor.extract_metadata_from_frame_text(ocr_text)
                is_sensitive = self.frame_metadata_extractor.is_sensitive_content(frame_metadata)
                
                if is_sensitive:
                    logger.warning(f"Detected sensitive data in frame {frame_path.name}: {frame_metadata}")
                    return True, frame_metadata
                
                return False, None
                
            except Exception as e:
                logger.error(f"Error in LLM metadata extraction: {e}")
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
            encoder_args = self._build_encoder_cmd('balanced')
            cmd = [
                'ffmpeg', '-nostdin', '-y', '-i', str(original_video),
                '-vf', vf,
                '-af', af,
                *encoder_args,
                '-movflags', '+faststart',
                str(output_video)
            ]
            
            logger.info(f"Re-encoding video without {len(frames_to_remove)} frames using {self.preferred_encoder['type']}")
            logger.debug(f"FFmpeg command with -nostdin: {' '.join(cmd)}")
            
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
                fallback_encoder_args = self._build_encoder_cmd('fast', fallback=True)
                cmd_no_audio = [
                    'ffmpeg', '-nostdin', '-y', '-i', str(original_video),
                    '-vf', vf,
                    '-an',  # No audio
                    *fallback_encoder_args,
                    str(output_video)
                ]
                result = subprocess.run(cmd_no_audio, capture_output=True, text=True, check=True)
                logger.info("Successfully re-encoded video without audio using CPU fallback")
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
                encoder_args = self._build_encoder_cmd('balanced')
                cmd = [
                    'ffmpeg', '-nostdin', '-y', '-i', str(input_video),
                    '-vf', crop_filter,
                    *encoder_args,
                    '-c:a', 'copy',  # Preserve audio
                    '-movflags', '+faststart',
                    str(output_video)
                ]
                logger.info(f"Using simple crop mask with {self.preferred_encoder['type']}: {crop_filter}")
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
                
                encoder_args = self._build_encoder_cmd('fast')
                cmd = [
                    'ffmpeg', '-nostdin', '-y', '-i', str(input_video),
                    '-vf', vf,
                    *encoder_args,
                    '-c:a', 'copy',  # Preserve audio
                    '-movflags', '+faststart',
                    str(output_video)
                ]
                logger.info(f"Using complex drawbox mask with {len(mask_filters)} regions using {self.preferred_encoder['type']}")
            
            logger.info(f"Applying mask to video: {input_video} -> {output_video}")
            logger.debug(f"FFmpeg masking command with -nostdin: {' '.join(cmd)}")
            
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

        Uses FrameOCR with preprocessing for better quality.
        Confidence is normalised to [0,1]. Empty-text frames are skipped.
        """
        for fp in frame_paths:
            # Load image and convert to numpy array
            img = Image.open(fp).convert('L')
            frame_array = np.array(img)
            
            # Use FrameOCR with preprocessing instead of raw pytesseract
            ocr_text, avg_conf, _ = self.frame_ocr.extract_text_from_frame(
                frame_array,
                roi=None,  # Could be enhanced with ROI if available
                high_quality=True
            )
            
            if not ocr_text.strip():
                continue  # nothing recognisable
            
            yield ocr_text, avg_conf
            
    def _iter_video(self, video_path: Path, total_frames: int) -> Iterator[Tuple[int, np.ndarray, int]]:
        """
        Yield (abs_frame_index, gray_frame, skip_value) with adaptive subsampling.
        
        Optimized for high-quality frame extraction to improve OCR accuracy.
        Sets OpenCV backend properties for maximum decode quality.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open %s", video_path)
            return
        
        # Set backend properties for higher quality frame decoding
        # CAP_PROP_FOURCC forces codec selection for quality
        # CAP_PROP_BUFFERSIZE ensures full frames are decoded
        try:
            # Try to set hardware decode preferences (may not work on all systems)
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except (AttributeError, cv2.error):
            # Not all OpenCV versions support this
            pass
        
        # Calculate adaptive skip for sampling
        skip = math.ceil(total_frames / 50)
        idx = 0
        
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            
            if idx % skip == 0:
                # Convert to grayscale for OCR processing
                # Use high-quality conversion algorithm
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                
                # Optional: Apply slight sharpening to compensate for video compression
                # This can help OCR by making text edges clearer
                gray = cv2.filter2D(gray, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
                
                yield idx, gray, skip
            idx += 1
            
        cap.release()

    def _update_video_sensitive_meta(self, video_file_obj, metadata: Dict[str, Any]):
        """
        Update VideoFile's SensitiveMeta with extracted metadata from frames.
        """
        try:
            #TODO should be in endoreg-db since this would cause a circular dependency @maxhild
            from endoreg_db.models import SensitiveMeta
            # Hol den OCR-Text, führe unified Extraktion aus:
            text = (metadata or {}).get('representative_ocr_text', '') or ''
            extracted = self.extract_metadata(text)  # LLM→spaCy

            # merge: extrahierte Felder überschreiben nur leere/Unknown
            merged = {**(metadata or {})}
            for k, v in (extracted or {}).items():
                if v not in (None, "", "Unknown"):
                    if not merged.get(k) or merged.get(k) in (None, "", "Unknown"):
                        merged[k] = v

            sensitive_meta, created = SensitiveMeta.objects.get_or_create(
                video_file=video_file_obj,
                defaults={
                    'patient_first_name': merged.get('patient_first_name'),
                    'patient_last_name': merged.get('patient_last_name'),
                    'patient_dob': merged.get('patient_dob'),
                    'casenumber': merged.get('casenumber'),
                    'patient_gender': merged.get('patient_gender'),
                    'examination_date': merged.get('examination_date'),
                    'examination_time': merged.get('examination_time'),
                    'examiner': merged.get('examiner'),
                    'representative_ocr_text': merged.get('representative_ocr_text', ''),
                }
            )

            if not created:
                updated = False
                for field in [
                    'patient_first_name','patient_last_name','patient_dob','casenumber',
                    'patient_gender','examination_date','examination_time','examiner',
                    'representative_ocr_text'
                ]:
                    nv = merged.get(field)
                    cv = getattr(sensitive_meta, field, None)
                    if nv and nv not in (None, "", "Unknown") and (not cv or cv in (None, "", "Unknown")):
                        setattr(sensitive_meta, field, nv)
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

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """LLM-first mit automatischem Modell-Fallback; spaCy als Notanker."""
        if not text or not text.strip():
            return {}
        
        logger.debug(f"Extracting metadata from text of length {len(text)} with content: {text}...")

        meta: Dict[str, Any] = {}
        # Nur versuchen, wenn LLM aktiviert und verfügbar ist
        if getattr(self, "use_llm", False) and getattr(self, "ollama_extractor", None) is not None:
            try:
                meta_obj = self.ollama_extractor.extract_metadata(text)  # Pydantic-Objekt oder None
                if meta_obj:
                    meta = meta_obj.model_dump()
            except Exception as e:
                logger.warning(f"Ollama extraction failed: {e}")
                meta = {}

        # LLM fehlgeschlagen/leer oder nicht aktiv → spaCy-Extractor fallback
        if not meta:
            try:
                if callable(self.PatientDataExtractor):
                    spacy_meta = self.PatientDataExtractor(text)
                elif hasattr(self.PatientDataExtractor, "extract"):
                    spacy_meta = self.PatientDataExtractor.extract(text)
                elif hasattr(self.PatientDataExtractor, "patient_extractor"):
                    spacy_meta = self.PatientDataExtractor.patient_extractor(text)
                else:
                    spacy_meta = {}
                if isinstance(spacy_meta, dict):
                    meta = spacy_meta
            except Exception as e:
                logger.error(f"spaCy fallback failed: {e}")
                meta = {}

        return meta or {}

    def extract_metadata_deepseek(self, text: str) -> Dict[str, Any]:
        """Extract metadata using the unified extractor (compatibility wrapper)."""
        return self.extract_metadata(text)

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
                    sensitive_frame_indices.append(frame_idx)
                    hits_count += 1
                    logger.info(f"Found sensitive data in frame {frame_idx}")
                    if len(ocr_text) < len(best_text):
                        best_text = ocr_text
                        logger.debug(f"OCR text: {ocr_text}")
                
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

    def _log_hf_cache_info(self) -> None:
            base = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
            hub  = Path(os.environ.get("HF_HUB_CACHE", str(base / "hub")))
            tfc  = Path(os.environ.get("TRANSFORMERS_CACHE", str(base)))
            candidates = [
                hub / "models--openbmb--MiniCPM-o-2_6",
                base / "models--openbmb--MiniCPM-o-2_6",
                tfc / "models--openbmb--MiniCPM-o-2_6",
            ]
            logger.info(f"HF_HOME={base} HF_HUB_CACHE={hub} TRANSFORMERS_CACHE={tfc}")
            for p in candidates:
                try:
                    size = shutil.disk_usage(p).total if p.exists() else 0
                except Exception:
                    size = 0
                logger.info(f"HF cache candidate: {p} exists={p.exists()} size_bytes={p.stat().st_size if p.exists() and p.is_file() else 'dir' if p.exists() else 0}")
    
    def _process_frame_enriched(
        self,
        gray_frame: np.ndarray,
        endoscope_roi: dict | None,
        frame_id: int | None = None,
        collect_for_batch: bool = True,
    ) -> tuple[bool, dict, str, float]:
        """
        Erweiterte Frame-Verarbeitung mit Sammlung für Batch-Metadaten-Extraktion.
        
        Args:
            gray_frame: Grayscale frame array
            endoscope_roi: ROI für Endoskop-Bereich
            frame_id: Frame-Index
            collect_for_batch: Ob Frame-Daten für Batch-Verarbeitung gesammelt werden sollen
            
        Returns:
            is_sensitive, frame_metadata, ocr_text, ocr_conf
        """
        logger.debug(f"Processing enriched frame_id={frame_id or 'unknown'}")
        
        # Intelligente Frame-Auswahl basierend auf FrameSamplingOptimizer
        should_process = True
        if collect_for_batch and frame_id is not None:
            should_process = self.frame_sampling_optimizer.should_process_frame(
                frame_id, self.current_video_total_frames
            )
        
        if not should_process:
            logger.debug(f"Skipping frame {frame_id} based on sampling optimization")
            return False, {}, "", 0.0
        
        # Normale Frame-Verarbeitung
        is_sensitive, frame_metadata, ocr_text, ocr_conf = self._process_frame(
            gray_frame, endoscope_roi, frame_id
        )
        
        # Sammle Frame-Daten für Batch-Verarbeitung
        if collect_for_batch and ocr_text and len(ocr_text.strip()) > 10:
            frame_data = {
                'frame_id': frame_id,
                'ocr_text': ocr_text,
                'ocr_confidence': ocr_conf,
                'is_sensitive': is_sensitive,
                'frame_metadata': frame_metadata,
                'frame_shape': gray_frame.shape,
                'timestamp': frame_id / 30.0 if frame_id else 0.0,  # Annahme: 30 FPS
            }
            
            self.frame_collection.append(frame_data)
            self.ocr_text_collection.append(ocr_text)
            
            # Registriere verarbeiteten Frame
            frame_hash = str(hash(ocr_text[:100]))  # Einfacher Hash für Duplikatserkennung
            self.frame_sampling_optimizer.register_processed_frame(frame_hash, frame_metadata)
            
            logger.debug(f"Collected frame {frame_id} for batch processing (total: {len(self.frame_collection)})")
        
        return is_sensitive, frame_metadata, ocr_text, ocr_conf
    
    def _extract_enriched_metadata_batch(self) -> Dict[str, Any]:
        """
        Extrahiert erweiterte Metadaten aus gesammelten Frame-Daten.
        
        Returns:
            Erweiterte Metadaten-Dictionary
        """
        if not self.frame_collection:
            logger.warning("Keine Frame-Daten für Batch-Extraktion gesammelt")
            return {}
        
        logger.info(f"Extracting enriched metadata from {len(self.frame_collection)} collected frames")
        
        try:
            # Verwende EnrichedMetadataExtractor für Multi-Frame-Analyse
            enriched_metadata = self.enriched_extractor.extract_from_frame_sequence(
                frames_data=self.frame_collection,
                ocr_texts=self.ocr_text_collection
            )
            
            logger.info(f"✅ Enriched metadata extraction successful: {len(enriched_metadata)} fields")
            
            # Zeige gefundene Daten
            if enriched_metadata.get('patient_name'):
                logger.info(f"Found patient: {enriched_metadata['patient_name']}")
            if enriched_metadata.get('patient_age'):
                logger.info(f"Found age: {enriched_metadata['patient_age']}")
            if enriched_metadata.get('examination_date'):
                logger.info(f"Found date: {enriched_metadata['examination_date']}")
            
            return enriched_metadata
            
        except Exception as e:
            logger.error(f"Enriched metadata extraction failed: {e}")
            logger.debug(f"Frame collection size: {len(self.frame_collection)}")
            logger.debug(f"OCR text collection size: {len(self.ocr_text_collection)}")
            return {}
    
    def _reset_frame_collection(self):
        """Setzt die Frame-Sammlung für ein neues Video zurück."""
        self.frame_collection.clear()
        self.ocr_text_collection.clear()
        logger.debug("Frame collection reset for new video")
    
    def _ocr_with_minicpm(self, gray_frame: np.ndarray) -> tuple[str, float, dict, bool]:
        """
        OCR mit MiniCPM LLM-basiertem Modell. Fallback auf TesserOCR bei Fehler.
        """
        try:
            pil_img = Image.fromarray(gray_frame, mode="L").convert("RGB")
            is_sensitive, frame_metadata, ocr_text = self.minicpm_ocr.detect_sensitivity_unified(
                pil_img, context="endoscopy video frame"
            )
            logger.debug(f"MiniCPM extracted keys: {sorted(frame_metadata.keys()) if isinstance(frame_metadata, dict) else type(frame_metadata).__name__}")
            return ocr_text, 1.0, frame_metadata, is_sensitive
        except ValueError as ve:
            logger.error(f"MiniCPM processing failed: {ve}")
            logger.warning("MiniCPM failed – falling back to TesserOCR")
            self.use_minicpm = False
            self.minicpm_ocr = None
            return self._ocr_with_tesserocr(gray_frame)
        except Exception as e:
            logger.exception(f"Unexpected MiniCPM error: {e}")
            return "", 0.0, {}, False

    def _ocr_with_tesserocr(self, gray_frame: np.ndarray, endoscope_roi: dict | None = None) -> tuple[str, float, dict, bool]:
        """
        OCR mit TesserOCR und Metadatenextraktion.
        """
        try:
            logger.debug("Using TesserOCR OCR engine")
            ocr_text, ocr_conf, _ = self.frame_ocr.extract_text_from_frame(
                gray_frame, roi=endoscope_roi, high_quality=True
            )
            logger.debug(f"TesserOCR extracted text length: {len(ocr_text or '')}, conf: {ocr_conf:.3f}")
            frame_metadata = (
                self.frame_metadata_extractor.extract_metadata_from_frame_text(ocr_text)
                if ocr_text else {}
            )
            is_sensitive = self.frame_metadata_extractor.is_sensitive_content(frame_metadata)
            if hasattr(self.best_frame_text, "push"):
                self.best_frame_text.push(ocr_text, ocr_conf)
            return ocr_text, ocr_conf, frame_metadata, is_sensitive
        except Exception as e:
            logger.exception(f"TesserOCR OCR failed: {e}")
            return "", 0.0, {}, False