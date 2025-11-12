"""
Frame-level anonymization module for video processing.

This module provides functionality to:
- Extract frames from videos using ffmpeg
- Apply specialized frame OCR to detect sensitive information
- Remove or mask frames containing sensitive data
- Re-encode cleaned videos

Uses specialized frame processing components separated from PDF logic.
"""

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from locale import normalize
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from numpy.f2py.symbolic import Op
from PIL import Image
from spacy.lang import en

from .frame_metadata_extractor import FrameMetadataExtractor
from .masking import MaskApplication
from .ocr_frame import FrameOCR

# from .ocr_minicpm import (
#     _can_load_model,
#     create_minicpm_ocr,
# )
from .ollama_llm_meta_extraction import (
    EnrichedMetadataExtractor,
    FrameSamplingOptimizer,
    OllamaOptimizedExtractor,
)
from .roi_processor import ROIProcessor
from .sensitive_meta_interface import SensitiveMeta
from .spacy_extractor import PatientDataExtractor
from .utils.ollama import ensure_ollama
from .utils.roi_normalization import normalize_roi_keys
from .video_encoder import VideoEncoder

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

    def __init__(
        self,
        use_minicpm: Optional[bool] = False,
        minicpm_config: Optional[Dict[str, Any]] = None,
        use_llm: Optional[bool] = False,
    ):
        # Initialize specialized frame processing components
        self.frame_ocr = FrameOCR()
        self.frame_metadata_extractor = FrameMetadataExtractor()
        self.patient_data_extractor = PatientDataExtractor()
        self.roi_processor = ROIProcessor()

        # Enhanced OCR integration - use enhanced components if available
        logger.info("Initializing with Enhanced OCR components (OCR_FIX_V1 enabled)")
        self.use_enhanced_ocr = True

        # LLM usage flag (guard)
        self.use_llm = bool(use_llm)

        # Initialize MiniCPM-o 2.6 if enabled
        # self.use_minicpm = use_minicpm
        # self.minicpm_ocr = create_minicpm_ocr() if use_minicpm else None
        # self._log_hf_cache_info()

        # Initialize the optimized ollama processing pipeline (guarded)
        self.ollama_proc = None
        self.ollama_extractor = None
        self.frame_sampling_optimizer = None
        self.enriched_extractor = None
        if self.use_llm:
            try:
                # Initialize Ollama for LLM processing
                self.ollama_proc = ensure_ollama()

                # Try to initialize OllamaOptimizedExtractor
                # If it fails (no models available), it will raise an exception caught below
                self.ollama_extractor = OllamaOptimizedExtractor()

                # Only initialize other components if ollama_extractor succeeded
                if self.ollama_extractor and self.ollama_extractor.current_model:
                    # Initialize enriched metadata extraction components
                    self.frame_sampling_optimizer = FrameSamplingOptimizer(max_frames=100, skip_similar_threshold=0.85)
                    self.enriched_extractor = EnrichedMetadataExtractor(
                        ollama_extractor=self.ollama_extractor,
                        frame_optimizer=self.frame_sampling_optimizer,
                    )
                else:
                    logger.warning("Ollama models not available, disabling LLM features")
                    self.use_llm = False
                    self.ollama_extractor = None

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

        # Hardware acceleration detection, Encoder setup
        self.video_encoder = VideoEncoder(mask_video_streaming=False, create_mask_config_from_roi=False)
        self.nvenc_available = self.video_encoder.nvenc_available
        self.preferred_encoder = self.video_encoder.preferred_encoder
        self._build_encoder_cmd = self.video_encoder._build_encoder_cmd

        # Masking
        self.mask_application = MaskApplication(self.preferred_encoder)
        self._mask_video_streaming = self.mask_application.mask_video_streaming
        self._create_mask_config_from_roi = self.mask_application.create_mask_config_from_roi

        # Sensitive metadata dictionary
        self.sensitive_meta: SensitiveMeta = SensitiveMeta()
        self.sensitive_meta_dict: Dict[str, Any] = self.sensitive_meta.to_dict()

        # ROI Processor - dict traversal for convenience
        self.roi_processor = ROIProcessor()

        logger.info(f"Hardware acceleration: NVENC {'available' if self.nvenc_available else 'not available'}")
        logger.info(f"Using encoder: {self.preferred_encoder}")

        # if self.use_minicpm:
        #     logger.warning("MiniCPM currently not functional; falling back to TesserOCR.")
        #     self.use_minicpm = False

        # try:
        #     minicpm_config = minicpm_config or {}
        #     if _can_load_model():
        #         self.minicpm_ocr = create_minicpm_ocr(**minicpm_config)
        #     else:
        #         logger.warning("Insufficient storage to load MiniCPM-o 2.6 model. Falling back to traditional OCR.")
        #         self.use_minicpm = False
        #         self.minicpm_ocr = None

        #     logger.info("MiniCPM-o 2.6 initialized successfully")
        # except Exception as e:
        #     logger.warning(f"Failed to initialize MiniCPM-o 2.6: {e}. Falling back to traditional OCR.")
        #     self.use_minicpm = False
        #     self.minicpm_ocr = None

    def clean_video(
        self,
        video_path: Path,
        endoscope_image_roi: Optional[dict[str, int]],
        endoscope_data_roi_nested: Optional[dict[str, dict[str, int | None]] | list | None],
        output_path: Optional[Path] = None,
        technique: str = "mask_overlay",
    ) -> tuple[Path, Dict[str, Any]]:
        """
        Handles the cleaning of a video by removing or masking frames with sensitive information.
        Args:
            video_path: Path to the input video file
            endoscope_image_roi: ROI for image masking
            endoscope_data_roi_nested: Nested ROI for data extraction
            output_path: Optional path for the output cleaned video
            technique: 'remove_frames' or 'mask_overlay'
            extended: Whether to perform extended metadata extraction

        Returns:
            A tuple containing the output video path and a dictionary of sensitive metadata.

        Refactored version: single code path, fewer duplicated branches. Batch Metadata logic preserves previous output based on confidence.
        """

        output_video = output_path or video_path.with_stem(f"{video_path.stem}_anony")
        accumulated: dict[str, Any] = {
            "file_path": str(object=video_path),
            "patient_first_name": None,
            "patient_last_name": None,
            "patient_dob": None,
            "casenumber": None,
            "patient_gender_name": None,
            "examination_date": None,
            "examination_time": None,
            "examiner_first_name": None,
            "examiner_last_name": None,
            "center": None,
            "representative_ocr_text": None,
            "source": "frame_extraction",
        }
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        long_video = total_frames > 10_000
        max_samples = min(500, total_frames // 20) if long_video else total_frames
        logger.info(
            "%s video detected (%d frames). Sampling ≤%d frames.",
            "Long" if long_video else "Short",
            total_frames,
            max_samples,
        )
        sensitive_idx: list[int] = []
        sampled = 0
        self.frame_collection = []  # Reset für neuen Batch
        for idx, gray_frame, stride in self._iter_video(video_path, total_frames):
            if sampled >= max_samples:
                break
            sampled += 1
            is_sensitive, frame_meta, ocr_text, ocr_conf = self._process_frame_single(
                gray_frame,
                endoscope_image_roi=endoscope_image_roi,
                endoscope_data_roi_nested=endoscope_data_roi_nested,
                frame_id=idx,
                collect_for_batch=True,
            )
            accumulated = self.frame_metadata_extractor.merge_metadata(accumulated, frame_meta)
            if is_sensitive:
                sensitive_idx.append(idx)
                self.sensitive_meta.safe_update(accumulated)

        
        # Batch-Metadaten-Anreicherung nach Frame-Loop
        if self.frame_collection:
            batch_enriched = self._extract_enriched_metadata_batch()
            if batch_enriched:
                accumulated = self.frame_metadata_extractor.merge_metadata(accumulated, batch_enriched)
        sensitive_ratio = len(sensitive_idx) / total_frames if total_frames else 0.0
        logger.info(
            "Sensitive frames: %d/%d (%.1f %%)",
            len(sensitive_idx),
            total_frames,
            100 * sensitive_ratio,
        )
        if technique == "remove_frames":
            logger.info("Using frame‑removal strategy.")
            ok = self.remove_frames_from_video_streaming(
                video_path,
                sensitive_idx,
                output_video,
                total_frames=total_frames,
            )
            if not ok:
                logger.error("Frame removal failed.")
        elif technique == "mask_overlay":
            logger.info("Using masking strategy.")
            if endoscope_image_roi and self._validate_roi(endoscope_image_roi):
                mask_cfg = self._create_mask_config_from_roi(endoscope_image_roi)
            else:
                mask_cfg = {"image_width": 1920, "image_height": 1080}
            self._mask_video_streaming(video_path, mask_cfg, output_video, use_named_pipe=True)

        # ----------------------- persist metadata, apply type checking---------------------------
        self.sensitive_meta.safe_update(accumulated)
        
        return output_video, self.sensitive_meta_dict

    # def _get_primary_ocr_engine(self) -> str:
    #     """Return the name of the primary OCR engine being used."""
    #     return "MiniCPM-o 2.6" if (self.use_minicpm and self.minicpm_ocr) else "FrameOCR + LLM"

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
                "ffprobe",
                "-nostdin",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            format_info = json.loads(result.stdout)

            # Extract key information for optimization
            video_stream = next((s for s in format_info["streams"] if s["codec_type"] == "video"), {})
            audio_streams = [s for s in format_info["streams"] if s["codec_type"] == "audio"]

            analysis = {
                "video_codec": video_stream.get("codec_name", "unknown"),
                "pixel_format": video_stream.get("pix_fmt", "unknown"),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "has_audio": len(audio_streams) > 0,
                "audio_codec": audio_streams[0].get("codec_name", "none") if audio_streams else "none",
                "container_format": format_info["format"].get("format_name", "unknown"),
                "duration": float(format_info["format"].get("duration", 0)),
                "size_bytes": int(format_info["format"].get("size", 0)),
                "can_stream_copy": self._can_use_stream_copy(video_stream, audio_streams),
            }

            logger.debug(f"Video format analysis: {analysis}")
            return analysis

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to analyze video format: {e}")
            return {
                "video_codec": "unknown",
                "pixel_format": "unknown",
                "width": 1920,
                "height": 1080,
                "has_audio": True,
                "audio_codec": "unknown",
                "container_format": "unknown",
                "duration": 0,
                "size_bytes": 0,
                "can_stream_copy": False,
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
        good_video_codecs = {"h264", "h265", "hevc", "vp8", "vp9", "av1"}
        good_audio_codecs = {"aac", "mp3", "opus", "vorbis"}

        video_codec = video_stream.get("codec_name", "").lower()

        # Check video codec compatibility
        if video_codec not in good_video_codecs:
            logger.debug(f"Video codec {video_codec} not suitable for stream copy")
            return False

        # Check audio codec compatibility
        for audio_stream in audio_streams:
            audio_codec = audio_stream.get("codec_name", "").lower()
            if audio_codec not in good_audio_codecs:
                logger.debug(f"Audio codec {audio_codec} not suitable for stream copy")
                return False

        # Check pixel format - some 10-bit formats need conversion
        pixel_format = video_stream.get("pix_fmt", "")
        if "10le" in pixel_format or "422" in pixel_format:
            logger.debug(f"Pixel format {pixel_format} may need conversion")
            return False

        return True

    @contextmanager
    def _named_pipe(self, suffix=".mp4"):
        temp_dir = Path(tempfile.mkdtemp(prefix="video_pipes_"))
        pipe_path = temp_dir / f"stream{suffix}"
        os.mkfifo(pipe_path)
        try:
            yield pipe_path
        finally:
            try:
                pipe_path.unlink()  # Make sure FIFO is removed
                temp_dir.rmdir()  # Ensure temporary directory is cleaned up
            except OSError as e:
                logger.warning(f"Failed to clean up temporary FIFO: {e}")


    def _stream_copy_with_pixel_conversion(
        self,
        input_video: Path,
        output_video: Path,
        target_pixel_format: str = "yuv420p",
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
            encoder_args = self._build_encoder_cmd("balanced")

            # Build FFmpeg command for pixel format conversion with hardware acceleration
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(input_video),
                "-vf",
                f"format={target_pixel_format}",  # Only convert pixel format
                *encoder_args,  # Use hardware-optimized encoder
                "-c:a",
                "copy",  # Copy audio without re-encoding
                "-avoid_negative_ts",
                "make_zero",  # Fix timestamp issues
                str(output_video),
            ]

            logger.info(f"Converting pixel format: {input_video} -> {output_video} (using {self.preferred_encoder['type']})")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"Pixel conversion output: {result.stderr}")

            return output_video.exists() and output_video.stat().st_size > 0

        except subprocess.CalledProcessError as e:
            # Try fallback encoder if hardware acceleration fails
            if self.preferred_encoder["type"] == "nvenc":
                logger.warning(f"NVENC pixel conversion failed, trying CPU fallback: {e.stderr}")
                return self._pixel_conversion_fallback(input_video, output_video, target_pixel_format)
            else:
                logger.error(f"Pixel format conversion failed: {e.stderr}")
                return False
        except Exception as e:
            logger.error(f"Pixel format conversion error: {e}")
            return False

    def _pixel_conversion_fallback(self, input_video: Path, output_video: Path, target_pixel_format: str) -> bool:
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
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(input_video),
                "-vf",
                f"format={target_pixel_format}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-c:a",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(output_video),
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
        format_info: Optional[Dict[str, Any]] = None,
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
            if format_info["can_stream_copy"]:
                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_video),
                    "-c",
                    "copy",  # Copy all streams without re-encoding
                    "-avoid_negative_ts",
                    "make_zero",
                    str(output_video),
                ]

                logger.info(f"Stream copying (no re-encoding): {input_video} -> {output_video}")

            else:
                # Need minimal conversion (usually just pixel format)
                pixel_fmt = format_info.get("pixel_format", "unknown")

                if "10le" in pixel_fmt or "422" in pixel_fmt:
                    logger.info(f"Converting {pixel_fmt} to yuv420p for compatibility")
                    return self._stream_copy_with_pixel_conversion(input_video, output_video)
                else:
                    # Use hardware-optimized encoding for unknown formats
                    encoder_args = self._build_encoder_cmd("fast")
                    cmd = [
                        "ffmpeg",
                        "-nostdin",
                        "-y",
                        "-i",
                        str(input_video),
                        *encoder_args,
                        "-c:a",
                        "copy",
                        str(output_video),
                    ]

                    logger.info(f"Fast re-encoding with {self.preferred_encoder['type']} and stream copy audio: {input_video} -> {output_video}")

            logger.debug(f"FFmpeg command with -nostdin: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"Stream copy output: {result.stderr}")
            return output_video.exists() and output_video.stat().st_size > 0

        except Exception as e:
            logger.error(f"Stream copy error: {e}")
            return False

    def remove_frames_from_video_streaming(
        self,
        original_video: Path,
        frames_to_remove: List[int],
        output_video: Path,
        total_frames: Optional[int] = None,
        use_named_pipe: bool = True,
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
            idx_list = "+".join([f"eq(n\\,{idx})" for idx in frames_to_remove])
            vf = f"select='not({idx_list})',setpts=N/FRAME_RATE/TB"
            af = f"aselect='not({idx_list})',asetpts=N/SR/TB"

            if use_named_pipe and len(frames_to_remove) < (total_frames or 1000) * 0.1:
                # Use named pipe for small frame removal operations
                with self._named_pipe() as pipe_path:
                    try:
                        # Pipeline: filter frames -> pipe -> stream copy to final output
                        filter_cmd = [
                            "ffmpeg",
                            "-nostdin",
                            "-y",
                            "-i",
                            str(original_video),
                            "-vf",
                            vf,
                            "-af",
                            af,
                            "-f",
                            "matroska",  # Use MKV for better streaming compatibility
                            str(pipe_path),
                        ]

                        copy_cmd = [
                            "ffmpeg",
                            "-nostdin",
                            "-y",
                            "-fflags",
                            "nobuffer",
                            "-i",
                            str(pipe_path),
                            "-c",
                            "copy",  # Stream copy from pipe
                            "-movflags",
                            "+faststart",
                            str(output_video),
                        ]

                        logger.info("Using named pipe for frame removal streaming (MKV container)")
                        logger.debug(f"Filter command with -nostdin: {' '.join(filter_cmd)}")
                        logger.debug(f"Copy command with -nostdin: {' '.join(copy_cmd)}")

                        # Start filter process in background
                        filter_proc = subprocess.Popen(filter_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        # Start copy process (blocks until complete)
                        subprocess.run(copy_cmd, capture_output=True, text=True, check=True)

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
                if format_info["can_stream_copy"] and format_info["has_audio"]:
                    # Use hardware-optimized encoding to preserve quality
                    encoder_args = self._build_encoder_cmd("balanced")
                    cmd = [
                        "ffmpeg",
                        "-nostdin",
                        "-y",
                        "-i",
                        str(original_video),
                        "-vf",
                        vf,
                        "-af",
                        af,
                        *encoder_args,  # Use hardware-optimized encoder
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",  # Re-encode audio with high quality
                        "-movflags",
                        "+faststart",
                        str(output_video),
                    ]
                else:
                    # Video-only or format needs re-encoding
                    encoder_args = self._build_encoder_cmd("balanced")
                    cmd = [
                        "ffmpeg",
                        "-nostdin",
                        "-y",
                        "-i",
                        str(original_video),
                        "-vf",
                        vf,
                        *encoder_args,  # Use hardware-optimized encoder
                        "-an" if not format_info["has_audio"] else "-af",
                        af if format_info["has_audio"] else "",
                        "-movflags",
                        "+faststart" if format_info["has_audio"] else "",
                        str(output_video),
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
                fallback_encoder_args = self._build_encoder_cmd("fast", fallback=True)
                cmd_no_audio = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(original_video),
                    "-vf",
                    vf,
                    "-an",  # No audio
                    *fallback_encoder_args,
                    str(output_video),
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
                meta_obj = self.ollama_extractor.extract_metadata(text)
                if meta_obj is not None:
                    self.sensitive_meta.safe_update(meta_obj)
                    meta = self.sensitive_meta_dict
                else:
                    meta = None
            except Exception:
                meta = None
        if not meta and self.patient_data_extractor:
            try:
                meta = self.patient_data_extractor(text)
                self.sensitive_meta.safe_update(meta)
                meta = self.sensitive_meta_dict
            except Exception:
                meta = None
        if not meta:
            out = self.frame_metadata_extractor.extract_metadata_from_frame_text(text)
            self.sensitive_meta.safe_update(out)
            meta = self.sensitive_meta_dict
        return meta

    def _process_frame_single(
        self,
        gray_frame: np.ndarray,
        endoscope_image_roi: Optional[dict | None],
        endoscope_data_roi_nested: Optional[dict[str, dict[str, int | None]] | list | None],
        frame_id: int | None = None,
        collect_for_batch: bool = False,
    ) -> tuple[bool, dict, str, float]:
        """
        Konsolidierte Einzel-Frame-Verarbeitung mit OCR, Metadaten und optionaler Batch-Sammlung.
        """
        logger.debug(f"Processing frame_id={frame_id or 'unknown'}")
        ocr_text, ocr_conf, frame_metadata, is_sensitive = None, 0.0, self.sensitive_meta_dict, False

        # possible OCR with MiniCPM-o 2.6
        # if self.use_minicpm and self.minicpm_ocr:
        #     ocr_text, ocr_conf, frame_metadata, is_sensitive = self._ocr_with_minicpm(gray_frame)
        # else:

        ocr_text, ocr_conf, frame_metadata, is_sensitive = self._ocr_with_tesserocr(gray_frame, endoscope_data_roi_nested)

        # Einheitliche Metadaten-Extraktion
        if ocr_text:
            meta_unified = self._unified_metadata_extract(ocr_text)
            frame_metadata = self.frame_metadata_extractor.merge_metadata(frame_metadata, meta_unified)
        self.sensitive_meta.safe_update(frame_metadata)
        frame_metadata = self.sensitive_meta_dict

            
        # Für Batch-Enrichment sammeln
        if collect_for_batch and ocr_text:
            self.frame_collection.append(
                {
                    "frame_id": frame_id,
                    "ocr_text": ocr_text,
                    "meta": frame_metadata,
                    "is_sensitive": is_sensitive,
                }
            )
        assert isinstance(frame_metadata, Dict)
        return is_sensitive, frame_metadata, ocr_text, ocr_conf



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
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "fps=1",  # Extract 1 frame per second (adjust as needed)
            "-q:v",
            "1",  # Highest quality (1-31, lower is better)
            "-pix_fmt",
            "rgb24",  # RGB colorspace for maximum quality
            str(output_dir / "frame_%04d.png"),  # PNG for lossless compression
        ]

        # Limit frames if specified
        if max_frames:
            cmd.insert(-1, "-frames:v")
            cmd.insert(-1, str(max_frames))

        try:
            logger.info(f"Extracting high-quality frames from {video_path} to {output_dir}")
            logger.debug(f"FFmpeg command with -nostdin: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"ffmpeg output: {result.stderr}")

            # Get list of created frame files (now PNG)
            frame_files = sorted(output_dir.glob("frame_*.png"))
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
            if image.mode != "L":
                image = image.convert("L")

            frame_array = np.array(image)

            # Use FrameOCR with preprocessing instead of raw pytesseract
            ocr_text, ocr_conf, _ = self.frame_ocr.extract_text_from_frame(
                frame_array,
                roi=None,  # Could be enhanced with ROI if available
                high_quality=True,
            )

            logger.debug(f"OCR confidence for {frame_path.name}: {ocr_conf:.3f} OCR Text:{ocr_text}")

            if not ocr_text.strip():
                logger.debug(f"No text detected in frame {frame_path.name}")
                return False

            logger.debug(f"OCR text from {frame_path.name} (conf={ocr_conf:.3f}): {ocr_text[:100]}...")

            # Use the same name detection logic as report_reader
            patient_info = report_reader.patient_extractor(ocr_text)

            # Check if sensitive information was found
            sensitive_fields = ["patient_first_name", "patient_last_name", "casenumber"]
            has_sensitive_data = any(patient_info.get(field) not in [None, "", "Unknown"] for field in sensitive_fields)

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

    def detect_sensitive_on_frame_extended(self, frame_path: Path, report_reader) -> "bool, Dict[str, Any]":
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
            if image.mode != "L":
                image = image.convert("L")

            frame_array = np.array(image)

            # Use FrameOCR with preprocessing instead of raw pytesseract
            ocr_text, ocr_conf, _ = self.frame_ocr.extract_text_from_frame(
                frame_array,
                roi=None,  # Could be enhanced with ROI if available
                high_quality=True,
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

                return False

            except Exception as e:
                logger.error(f"Error in LLM metadata extraction: {e}")
                return False, None

        except Exception as e:
            logger.error(f"Error processing frame {frame_path}: {e}")
            # Fail-safe: if OCR crashes, keep the frame (better none deleted than all lost)
            return False, None

    def video_ocr_stream(self, frame_paths: List[Path]) -> Generator[tuple[str, float], Any, None]:
        """
        Yield (ocr_text, avg_confidence) for every frame in frame_paths.

        Uses FrameOCR with preprocessing for better quality.
        Confidence is normalised to [0,1]. Empty-text frames are skipped.
        """
        for fp in frame_paths:
            # Load image and convert to numpy array
            img = Image.open(fp).convert("L")
            frame_array = np.array(img)

            # Use FrameOCR with preprocessing instead of raw pytesseract
            ocr_text, avg_conf, _ = self.frame_ocr.extract_text_from_frame(
                frame_array,
                roi=None,  # Could be enhanced with ROI if available
                high_quality=True,
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
        skip = max(1, math.ceil(total_frames / 50))
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
                gray = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

                yield idx, gray, skip
            idx += 1

        cap.release()

    def _update_sensitive_meta_dict(self, metadata: Dict[str, Any]):
        """
        Update SensitiveMeta interface with extracted metadata from frames.
        """
        try:
            # Hol den OCR-Text, führe unified Extraktion aus:
            text = (metadata or {}).get("representative_ocr_text", "") or ""
            extracted = self.extract_metadata(text)  # LLM→spaCy

            # merge: extrahierte Felder überschreiben nur leere/Unknown
            merged = {**(metadata or {})}
            for k, v in (extracted or {}).items():
                if v not in (None, "", "Unknown"):
                    if not merged.get(k) or merged.get(k) in (None, "", "Unknown"):
                        merged[k] = v

            data = {
                "patient_first_name": merged.get("patient_first_name"),
                "patient_last_name": merged.get("patient_last_name"),
                "patient_dob": merged.get("patient_dob"),
                "casenumber": merged.get("casenumber"),
                "patient_gender_name": merged.get("patient_gender_name"),
                "examination_date": merged.get("examination_date"),
                "examination_time": merged.get("examination_time"),
                "examiner_first_name": merged.get("examiner_first_name"),
                "examiner_last_name": merged.get("examiner_last_name"),
                "center": merged.get("center"),
            }

            # Default center fallback
            if not data.get("center") or data["center"] in (None, "", "Unknown"):
                default_center = os.environ.get("DEFAULT_CENTER", "Endoscopy Center")
                data["center"] = default_center
                logger.debug(f"No center detected — using default: {default_center}")

            new_meta = SensitiveMeta.from_dict(data)
            if new_meta != self.sensitive_meta:
                self.sensitive_meta = new_meta
                logger.info("SensitiveMeta updated")

        except Exception as e:
            logger.error(f"Failed to update video sensitive metadata: {e}")

    def _safe_conf_list(self, raw_conf):
        """Convert values from data['conf'] to int >= 0 safely"""
        confs = []
        for c in raw_conf:
            try:
                conf_int = int(c)  # works for both str AND int
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
                meta_obj = self.ollama_extractor.extract_metadata(text)  # type: ignore  # Pydantic-Objekt oder None
                if meta_obj:
                    meta = meta_obj.model_dump()
            except Exception as e:
                logger.warning(f"Ollama extraction failed: {e}")
                meta = {}

        # LLM fehlgeschlagen/leer oder nicht aktiv → spaCy-Extractor fallback
        if meta == {}:
            try:
                if callable(self.patient_data_extractor):
                    spacy_meta = self.patient_data_extractor(text)
                elif hasattr(self.patient_data_extractor, "extract"):
                    spacy_meta = self.patient_data_extractor.extract(text)
                elif hasattr(self.patient_data_extractor, "patient_extractor"):
                    spacy_meta = self.patient_data_extractor.patient_extractor(text)
                else:
                    spacy_meta = {}
                if isinstance(spacy_meta, dict):
                    meta = spacy_meta
            except Exception as e:
                logger.error(f"spaCy fallback failed: {e}")
                meta = {}

        return meta or {}

    def _log_hf_cache_info(self) -> None:
        base = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        hub = Path(os.environ.get("HF_HUB_CACHE", str(base / "hub")))
        tfc = Path(os.environ.get("TRANSFORMERS_CACHE", str(base)))
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
            logger.info(
                f"HF cache candidate: {p} exists={p.exists()} size_bytes={p.stat().st_size if p.exists() and p.is_file() else 'dir' if p.exists() else 0}"
            )

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
        # Verwende EnrichedMetadataExtractor für Multi-Frame-Analyse
        if not self.frame_sampling_optimizer:
            self.frame_sampling_optimizer = FrameSamplingOptimizer()
        if not self.ollama_extractor:
            self.ollama_extractor = OllamaOptimizedExtractor()
        if not self.enriched_extractor:
            self.enriched_extractor = EnrichedMetadataExtractor(ollama_extractor=self.ollama_extractor, frame_optimizer=self.frame_sampling_optimizer)
        enriched_metadata = self.enriched_extractor.extract_from_frame_sequence(frames_data=self.frame_collection, ocr_texts=self.ocr_text_collection)

        logger.info(f"✅ Enriched metadata extraction successful: {len(enriched_metadata)} fields")

        return enriched_metadata

    def _reset_frame_collection(self):
        """Setzt die Frame-Sammlung für ein neues Video zurück."""
        self.frame_collection.clear()
        self.ocr_text_collection.clear()
        logger.debug("Frame collection reset for new video")

    def _ocr_with_tesserocr(
        self,
        gray_frame: np.ndarray,
        endoscope_data_roi_nested: Optional[dict[str, dict[str, int | None]] | list | None],
    ) -> tuple[str, float, dict, bool]:
        """
        OCR with TesserOCR and metadata extraction.
        Handles both dict-based and list-based ROI structures gracefully.
        Includes enhanced validation to filter gibberish output.
        """
        try:
            logger.debug("Using TesserOCR OCR engine with enhanced validation")
            frame_metadata: Dict[str, Any] = {}
            ocr_text = ""
            valid_texts = []  # Store only validated text

            # --- Normalize input ROI structure ---
            rois: list[dict[str, int | None]] = []

            if not endoscope_data_roi_nested:
                has_roi = False
            elif isinstance(endoscope_data_roi_nested, dict):
                # Original expected format
                rois = list(endoscope_data_roi_nested.values())
                has_roi = True
            elif isinstance(endoscope_data_roi_nested, list):
                # Flatten nested lists of dicts
                for item in endoscope_data_roi_nested:
                    if isinstance(item, dict):
                        rois.append(item)
                    elif isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, dict):
                                rois.append(sub)
                has_roi = len(rois) > 0
            else:
                logger.warning(f"Unexpected ROI type: {type(endoscope_data_roi_nested)}")
                has_roi = False

            # --- Run OCR ---
            if not has_roi:
                ocr_text, ocr_conf, _ = self.frame_ocr.extract_text_from_frame(gray_frame, roi={}, high_quality=True)

                # Validate full-frame OCR
                if self._is_valid_ocr_text(ocr_text):
                    valid_texts.append(ocr_text)
                else:
                    logger.debug(f"Full-frame OCR produced invalid text, filtering: {ocr_text[:100]}")
                    ocr_text = ""
            else:
                ocr_conf = 0.0
                for i, roi in enumerate(rois):
                    output, conf, _ = self.frame_ocr.extract_text_from_frame(gray_frame, roi=roi, high_quality=True)

                    # Validate each ROI's OCR output
                    if self._is_valid_ocr_text(output):
                        valid_texts.append(output)
                        frame_metadata[f"roi_{i}"] = output
                        ocr_conf = max(ocr_conf, conf)
                    else:
                        logger.debug(f"ROI {i} produced invalid text (filtered): {output[:50]}")
                        frame_metadata[f"roi_{i}"] = ""

                # Combine only valid texts
                ocr_text = "\n".join(valid_texts)

            logger.debug(f"TesserOCR extracted {len(valid_texts)} valid text regions, total length: {len(ocr_text)}, conf: {ocr_conf:.3f}")

            # --- Metadata Extraction ---
            # Always try to extract metadata from the combined OCR text
            if ocr_text:
                logger.debug("Extracting metadata from combined OCR text")
                extracted_meta = self.frame_metadata_extractor.extract_metadata_from_frame_text(ocr_text)
                # Merge extracted metadata with ROI metadata
                frame_metadata.update(extracted_meta)

            is_sensitive = self.frame_metadata_extractor.is_sensitive_content(frame_metadata)
            return ocr_text, ocr_conf, frame_metadata, is_sensitive

        except Exception as e:
            logger.exception(f"TesserOCR OCR failed: {e}")
            return "", 0.0, {}, False

    def _is_valid_ocr_text(self, text: str, min_alpha_ratio: float = 0.20, min_length: int = 3) -> bool:
        """
        Validate OCR text to filter out gibberish.

        Args:
            text: OCR extracted text
            min_alpha_ratio: Minimum ratio of alphabetic characters (default 0.20, relaxed from 0.35)
            min_length: Minimum text length (default 3)

        Returns:
            True if text appears valid, False if likely gibberish
        """
        if not text or len(text.strip()) < min_length:
            return False

        # Special case: Allow date/time patterns (have few letters but are valid)
        # Examples: "09:53:32", "2024-01-15", "15702/2024", "E 15702/2024809:53:32"
        import re

        # Time patterns: HH:MM:SS or HH:MM
        time_pattern = r"\d{1,2}:\d{2}(?::\d{2})?"
        # Date patterns: YYYY-MM-DD, DD.MM.YYYY, YYYY/MM/DD
        date_pattern = r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]\d{4}"
        # Case number patterns: E 12345/2024 or similar
        case_pattern = r"[A-Z]\s*\d{4,}/\d{4}"
        # Device ID patterns: long numbers with optional separators
        device_pattern = r"\d{8,}"

        if re.search(time_pattern, text) or re.search(date_pattern, text) or re.search(case_pattern, text) or re.search(device_pattern, text):
            # Contains structured data patterns - likely valid
            return True

        # Calculate alphabetic character ratio
        alpha_count = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_count / len(text)

        # Relaxed from 0.35 to 0.20 to allow more numeric/mixed content
        if alpha_ratio < min_alpha_ratio:
            return False

        # Check for excessive special characters
        expected_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß0123456789 .,:;/-()[]")
        nonstandard = sum(1 for c in text if c not in expected_chars)

        # Relaxed from 0.3 to 0.4 to allow more special characters
        if nonstandard > 0.4 * len(text):
            return False

        # Check for reasonable word structure
        words = [w for w in text.split() if len(w) > 1]
        if not words:
            return False

        # Most words should have vowels (German/English)
        vowels = set("aeiouäöüAEIOUÄÖÜ")
        words_with_vowels = sum(1 for word in words if any(c in vowels for c in word))

        # Relaxed from 0.25 to 0.15 to allow more abbreviations/codes
        if len(words) > 2 and words_with_vowels < 0.15 * len(words):
            return False

        return True

    def remove_frames_from_video(
        self,
        original_video: Path,
        frames_to_remove: List[int],
        output_video: Path,
        total_frames: Optional[int] = None,
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
            idx_list = "+".join([f"eq(n\\,{idx})" for idx in frames_to_remove])

            # Build video filter: select frames NOT in the removal list
            vf = f"select='not({idx_list})',setpts=N/FRAME_RATE/TB"

            # Build audio filter: keep audio in sync (or skip if no audio needed)
            af = f"aselect='not({idx_list})',asetpts=N/SR/TB"

            # Build ffmpeg command with properly quoted filters
            encoder_args = self._build_encoder_cmd("balanced")
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(original_video),
                "-vf",
                vf,
                "-af",
                af,
                *encoder_args,
                "-movflags",
                "+faststart",
                str(output_video),
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
                fallback_encoder_args = self._build_encoder_cmd("fast", fallback=True)
                cmd_no_audio = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(original_video),
                    "-vf",
                    vf,
                    "-an",  # No audio
                    *fallback_encoder_args,
                    str(output_video),
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
        masks_dir = Path(__file__).parent / "masks"
        mask_file = masks_dir / f"{device_name}_mask.json"
        stub = {
            "image_width": 1920,
            "image_height": 1080,
            "x": 550,
            "y": 0,
            "width": 1350,
            "height": 1080,
        }

        try:
            with mask_file.open() as f:
                return json.load(f)  # works if file is valid
        except (FileNotFoundError, json.JSONDecodeError):
            # create or overwrite with a fresh stub
            masks_dir.mkdir(parents=True, exist_ok=True)
            with mask_file.open("w") as f:
                json.dump(stub, f, indent=2)
            logger.warning(
                "Created or repaired mask file %s – please verify coordinates.",
                mask_file,
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
            if mask_config == {}:
                load_mask_config = self._load_mask(device_name="default")
                normalized = normalize_roi_keys(load_mask_config)
                mask_config = normalized if normalized is not None else {}
                logger.info("Using default mask configuration")

            normalized = normalize_roi_keys(mask_config)
            if normalized is not None:
                mask_config = normalized

            endoscope_x = mask_config.get("x")
            endoscope_y = mask_config.get("y")
            endoscope_w = mask_config.get("width")
            endoscope_h = mask_config.get("height")

            # Check if we can use simple crop (left strip masking)
            if endoscope_y == 0 and endoscope_h == mask_config.get("image_height", 1080):
                # Simple left crop case - crop everything to the right of the endoscope area
                crop_filter = f"crop=in_w-{endoscope_x}:in_h:{endoscope_x}:0"
                encoder_args = self._build_encoder_cmd("balanced")
                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_video),
                    "-vf",
                    crop_filter,
                    *encoder_args,
                    "-c:a",
                    "copy",  # Preserve audio
                    "-movflags",
                    "+faststart",
                    str(output_video),
                ]
                logger.info(f"Using simple crop mask with {self.preferred_encoder['type']}: {crop_filter}")
            else:
                # Complex masking using drawbox to black out sensitive areas
                # Mask everything except the endoscope area
                mask_filters = []
                assert endoscope_x is not None and endoscope_y is not None and endoscope_w is not None and endoscope_h is not None, (
                    "Mask configuration is incomplete."
                )

                # Left rectangle (0 to endoscope_x)
                if endoscope_x > 0:
                    mask_filters.append(f"drawbox=0:0:{endoscope_x}:{mask_config.get('image_height', 1080)}:color=black@1:t=fill")

                # Right rectangle (endoscope_x + endoscope_w to image_width)
                right_start = endoscope_x + endoscope_w
                image_width = mask_config.get("image_width", 1920)
                if right_start < image_width:
                    right_width = image_width - right_start
                    mask_filters.append(f"drawbox={right_start}:0:{right_width}:{mask_config.get('image_height', 1080)}:color=black@1:t=fill")

                # Top rectangle (within endoscope x range, 0 to endoscope_y)
                if endoscope_y > 0:
                    mask_filters.append(f"drawbox={endoscope_x}:0:{endoscope_w}:{endoscope_y}:color=black@1:t=fill")

                # Bottom rectangle (within endoscope x range, endoscope_y + endoscope_h to image_height)
                bottom_start = endoscope_y + endoscope_h
                image_height = mask_config.get("image_height", 1080)
                if bottom_start < image_height:
                    bottom_height = image_height - bottom_start
                    mask_filters.append(f"drawbox={endoscope_x}:{bottom_start}:{endoscope_w}:{bottom_height}:color=black@1:t=fill")

                # Combine all mask filters
                vf = ",".join(mask_filters)

                encoder_args = self._build_encoder_cmd("fast")
                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(input_video),
                    "-vf",
                    vf,
                    *encoder_args,
                    "-c:a",
                    "copy",  # Preserve audio
                    "-movflags",
                    "+faststart",
                    str(output_video),
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

        normalized = normalize_roi_keys(roi)
        if not normalized:
            return False

        roi = normalized

        # Check for reasonable values (non-negative, not too large)
        try:
            x, y, width, height = roi["x"], roi["y"], roi["width"], roi["height"]

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

    # def _ocr_with_minicpm(self, gray_frame: np.ndarray) -> tuple[str, float, dict, bool]:
    #     """
    #     OCR mit MiniCPM LLM-basiertem Modell. Fallback auf TesserOCR bei Fehler.
    #     """
    #     try:
    #         pil_img = Image.fromarray(gray_frame, mode="L").convert("RGB")
    #         is_sensitive, frame_metadata, ocr_text = self.minicpm_ocr.detect_sensitivity_unified(pil_img, context="endoscopy video frame")
    #         logger.debug(f"MiniCPM extracted keys: {sorted(frame_metadata.keys()) if isinstance(frame_metadata, dict) else type(frame_metadata).__name__}")
    #         return ocr_text, 1.0, frame_metadata, is_sensitive
    #     except ValueError as ve:
    #         logger.error(f"MiniCPM processing failed: {ve}")
    #         logger.warning("MiniCPM failed – falling back to TesserOCR")
    #         self.use_minicpm = False
    #         self.minicpm_ocr = None
    #         return self._ocr_with_tesserocr(gray_frame)
    #     except Exception as e:
    #         logger.exception(f"Unexpected MiniCPM error: {e}")
    #         return "", 0.0, {}, False
