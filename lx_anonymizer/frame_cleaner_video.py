import logging
import math
import subprocess
import tempfile
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Iterator, List, Optional, Protocol, Tuple, cast

import cv2
import numpy as np
from lx_dtypes.models.meta.VideoMeta import (
    FrameRemovalFilterArgs,
    FrameRemovalPlan,
    VideoFormatProbe,
    VideoRoiBox,
)
from lx_dtypes.models.contracts.video_processing import VideoEncoderConfig

from lx_anonymizer.config import settings
from lx_anonymizer.utils.roi_normalization import normalize_roi_keys
from lx_anonymizer.video_processing import video_utils
from lx_anonymizer.video_processing.video_encoder import VideoEncoder
from lx_anonymizer.video_processing.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class _VideoCaptureProtocol(Protocol):
    def isOpened(self) -> bool: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def get(self, prop_id: int) -> float: ...

    def read(self) -> tuple[bool, np.ndarray]: ...

    def release(self) -> None: ...


class FrameCleanerVideoMixin:
    video_encoder: VideoEncoder
    video_processor: VideoProcessor
    preferred_encoder: VideoEncoderConfig
    build_encoder_cmd: Callable[..., List[str]]

    def _target_sample_count(self, total_frames: int) -> int:
        raise NotImplementedError

    def remove_frames_from_video_streaming(
        self,
        original_video: Path,
        frames_to_remove: List[int],
        output_video: Path,
        total_frames: Optional[int] = None,
        use_named_pipe: bool = True,
    ) -> bool:
        plan = self._build_frame_removal_plan(
            original_video=original_video,
            frames_to_remove=frames_to_remove,
            output_video=output_video,
            total_frames=total_frames,
            use_named_pipe=use_named_pipe,
        )
        filters = FrameRemovalFilterArgs()

        try:
            if not plan.frames_to_remove:
                logger.info("No frames to remove, using stream copy")
                return self.video_processor.stream_copy_video(
                    plan.original_video,
                    plan.output_video,
                )

            filters = self._remove_frames_streaming_impl(plan)
            return self._verify_video_output(plan.output_video)

        except subprocess.CalledProcessError as exc:
            return self._handle_frame_removal_process_error(plan, filters, exc)

        except subprocess.TimeoutExpired as exc:
            logger.error("Streaming frame removal timed out after %ss", exc.timeout)
            return False

        except Exception as exc:
            logger.error("Streaming frame removal error: %s", exc)
            return False

        finally:
            self._cleanup_filter_scripts(filters.filter_script_paths)

    def _remove_frames_streaming_impl(
        self,
        plan: FrameRemovalPlan,
    ) -> FrameRemovalFilterArgs:
        logger.info(
            "Removing %d frames using streaming method",
            len(plan.frames_to_remove),
        )
        filters = self._build_frame_removal_filters(plan.frames_to_remove)
        self._execute_frame_removal(plan=plan, filters=filters)
        return filters

    @staticmethod
    def _build_frame_removal_plan(
        *,
        original_video: Path,
        frames_to_remove: List[int],
        output_video: Path,
        total_frames: Optional[int],
        use_named_pipe: bool,
    ) -> FrameRemovalPlan:
        return FrameRemovalPlan(
            original_video=original_video,
            frames_to_remove=frames_to_remove,
            output_video=output_video,
            total_frames=total_frames,
            use_named_pipe=use_named_pipe,
            ffmpeg_timeout=max(10, int(getattr(settings, "LLM_TIMEOUT", 30)) * 10),
        )

    def _build_frame_removal_filters(
        self, frames_to_remove: list[int]
    ) -> FrameRemovalFilterArgs:
        vf, af = self._build_frame_drop_filters(frames_to_remove)
        vf_args, af_args, filter_script_paths = self._build_filter_args(vf, af)
        return FrameRemovalFilterArgs(
            vf_args=vf_args,
            af_args=af_args,
            filter_script_paths=filter_script_paths,
        )

    def _execute_frame_removal(
        self,
        *,
        plan: FrameRemovalPlan,
        filters: FrameRemovalFilterArgs,
    ) -> None:
        video_format = VideoFormatProbe.model_validate(
            video_utils.detect_video_format(plan.original_video)
        )
        if plan.should_use_named_pipe:
            self._remove_frames_via_named_pipe(
                original_video=plan.original_video,
                output_video=plan.output_video,
                vf_args=filters.vf_args,
                af_args=filters.af_args,
                has_audio=video_format.has_audio,
                ffmpeg_timeout=plan.ffmpeg_timeout,
            )
            return

        self._remove_frames_direct(
            original_video=plan.original_video,
            output_video=plan.output_video,
            vf_args=filters.vf_args,
            af_args=filters.af_args,
            has_audio=video_format.has_audio,
            ffmpeg_timeout=plan.ffmpeg_timeout,
        )

    def _handle_frame_removal_process_error(
        self,
        plan: FrameRemovalPlan,
        filters: FrameRemovalFilterArgs,
        exc: subprocess.CalledProcessError,
    ) -> bool:
        logger.error("Streaming frame removal failed: %s", exc.stderr)
        return self._remove_frames_cpu_fallback(
            original_video=plan.original_video,
            output_video=plan.output_video,
            vf_args=filters.vf_args,
            ffmpeg_timeout=plan.ffmpeg_timeout,
        )

    @staticmethod
    def _cleanup_filter_scripts(filter_script_paths: list[Path]) -> None:
        for script_path in filter_script_paths:
            try:
                script_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove temporary filter script %s", script_path)

    def _remove_frames_via_named_pipe(
        self,
        original_video: Path,
        output_video: Path,
        vf_args: List[str],
        af_args: List[str],
        has_audio: bool,
        ffmpeg_timeout: int,
    ) -> None:
        """
        Remove frames via a named-pipe streaming pipeline:
        filter -> mkv pipe -> stream copy to final output.
        """
        with video_utils.named_pipe() as pipe_path:
            filter_proc: Optional[subprocess.Popen[bytes]] = None
            try:
                filter_cmd = self._named_pipe_filter_cmd(
                    original_video=original_video,
                    pipe_path=pipe_path,
                    vf_args=vf_args,
                    af_args=af_args,
                    has_audio=has_audio,
                )
                copy_cmd = self._named_pipe_copy_cmd(
                    pipe_path=pipe_path,
                    output_video=output_video,
                )

                self._log_named_pipe_commands(filter_cmd, copy_cmd)
                filter_proc = self._start_filter_process(filter_cmd)
                self._run_checked(copy_cmd, ffmpeg_timeout)
                self._wait_checked(filter_proc, filter_cmd, ffmpeg_timeout)

            except subprocess.TimeoutExpired as exc:
                logger.error(
                    "Named-pipe frame removal timed out after %ss", exc.timeout
                )
                raise
            finally:
                self._terminate_filter_process(filter_proc)

    @staticmethod
    def _named_pipe_filter_cmd(
        *,
        original_video: Path,
        pipe_path: Path,
        vf_args: list[str],
        af_args: list[str],
        has_audio: bool,
    ) -> list[str]:
        cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(original_video), *vf_args]
        if has_audio:
            cmd.extend(af_args)
        cmd.extend(["-f", "matroska", str(pipe_path)])
        return cmd

    @staticmethod
    def _named_pipe_copy_cmd(*, pipe_path: Path, output_video: Path) -> list[str]:
        return [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-fflags",
            "nobuffer",
            "-i",
            str(pipe_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_video),
        ]

    @staticmethod
    def _log_named_pipe_commands(filter_cmd: list[str], copy_cmd: list[str]) -> None:
        logger.info("Using named pipe for frame removal streaming (MKV container)")
        logger.debug("Filter command with -nostdin: %s", " ".join(filter_cmd))
        logger.debug("Copy command with -nostdin: %s", " ".join(copy_cmd))

    @staticmethod
    def _start_filter_process(filter_cmd: list[str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            filter_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _run_checked(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )
        return result

    @staticmethod
    def _wait_checked(
        process: subprocess.Popen[bytes],
        cmd: list[str],
        timeout: int,
    ) -> None:
        return_code = process.wait(timeout=timeout)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        logger.debug("Streaming frame removal completed via named pipe")

    @staticmethod
    def _terminate_filter_process(
        filter_proc: Optional[subprocess.Popen[bytes]],
    ) -> None:
        if filter_proc is None or filter_proc.poll() is not None:
            return
        filter_proc.terminate()
        try:
            filter_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            filter_proc.kill()
            filter_proc.wait(timeout=10)

    def _remove_frames_direct(
        self,
        original_video: Path,
        output_video: Path,
        vf_args: List[str],
        af_args: List[str],
        has_audio: bool,
        ffmpeg_timeout: int,
    ) -> None:
        """
        Remove frames via direct ffmpeg processing.
        """
        cmd = self._direct_frame_removal_cmd(
            original_video=original_video,
            output_video=output_video,
            vf_args=vf_args,
            af_args=af_args,
            has_audio=has_audio,
        )

        logger.info(
            "Direct frame removal processing using %s",
            self.preferred_encoder.type,
        )
        logger.debug("FFmpeg command with -nostdin: %s", " ".join(cmd))

        result = self._run_checked(cmd, ffmpeg_timeout)
        logger.debug("Direct frame removal output: %s", result.stderr)

    def _direct_frame_removal_cmd(
        self,
        *,
        original_video: Path,
        output_video: Path,
        vf_args: list[str],
        af_args: list[str],
        has_audio: bool,
    ) -> list[str]:
        encoder_args = self.build_encoder_cmd("balanced")
        cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(original_video), *vf_args]
        if has_audio:
            cmd.extend(
                [
                    *af_args,
                    *encoder_args,
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    str(output_video),
                ]
            )
            return cmd

        cmd.extend(
            [
                *encoder_args,
                "-an",
                str(output_video),
            ]
        )
        return cmd

    def _remove_frames_cpu_fallback(
        self,
        original_video: Path,
        output_video: Path,
        vf_args: List[str],
        ffmpeg_timeout: int,
    ) -> bool:
        """
        Retry frame removal without audio using CPU fallback.
        """
        try:
            logger.warning(
                "Retrying frame removal without audio processing using CPU..."
            )
            cmd_no_audio = self._cpu_fallback_frame_removal_cmd(
                original_video=original_video,
                output_video=output_video,
                vf_args=vf_args,
            )
            self._run_checked(cmd_no_audio, ffmpeg_timeout)

            logger.info("Successfully removed frames without audio using CPU fallback")
            return self._verify_video_output(output_video)

        except subprocess.TimeoutExpired as exc:
            logger.error(
                "Frame removal CPU fallback timed out after %ss",
                exc.timeout,
            )
            return False
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Frame removal CPU fallback also failed: %s",
                exc.stderr,
            )
            return False

    def _cpu_fallback_frame_removal_cmd(
        self,
        *,
        original_video: Path,
        output_video: Path,
        vf_args: list[str],
    ) -> list[str]:
        return [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(original_video),
            *vf_args,
            "-an",
            *self.build_encoder_cmd("fast", fallback=True),
            str(output_video),
        ]

    @staticmethod
    def _verify_video_output(output_video: Path) -> bool:
        if output_video.exists() and output_video.stat().st_size > 0:
            logger.info("Successfully removed frames: %s", output_video)
            return True

        logger.error("Frame removal output is empty or missing")
        return False

    def _iter_video(
        self, video_path: Path, total_frames: int
    ) -> Iterator[Tuple[int, np.ndarray, int]]:
        cap = cast(_VideoCaptureProtocol, cv2.VideoCapture(str(video_path)))
        if not cap.isOpened():
            logger.error("Cannot open %s", video_path)
            return

        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except (AttributeError, cv2.error):
            pass

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_samples = self._target_sample_count(total_frames) or 1
        calculated_skip = (
            math.ceil(total_frames / target_samples) if total_frames else 1
        )

        max_skip_limit = int(fps * 2)
        skip = max(5, min(calculated_skip, max_skip_limit))

        idx = 0

        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if idx % skip == 0:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                yield idx, gray, skip
            idx += 1

        cap.release()

    @staticmethod
    def _frame_ranges(indices: List[int]) -> List[Tuple[int, int]]:
        clean = sorted({int(idx) for idx in indices if int(idx) >= 0})
        if not clean:
            return []
        ranges: List[Tuple[int, int]] = []
        start = clean[0]
        end = clean[0]
        for value in clean[1:]:
            if value == end + 1:
                end = value
                continue
            ranges.append((start, end))
            start = value
            end = value
        ranges.append((start, end))
        return ranges

    def _build_frame_drop_filters(self, frames_to_remove: List[int]) -> Tuple[str, str]:
        clean = sorted({int(idx) for idx in frames_to_remove if int(idx) >= 0})
        if not clean:
            return "select='1',setpts=N/FRAME_RATE/TB", "aselect='1',asetpts=N/SR/TB"

        terms: List[str] = []
        if len(clean) <= 64:
            terms = [f"eq(n\\,{idx})" for idx in clean]
        else:
            ranges = self._frame_ranges(clean)
            for start, end in ranges:
                if start == end:
                    terms.append(f"eq(n\\,{start})")
                else:
                    terms.append(f"between(n\\,{start}\\,{end})")

        condition = "+".join(terms)
        vf = f"select='not({condition})',setpts=N/FRAME_RATE/TB"
        af = f"aselect='not({condition})',asetpts=N/SR/TB"
        return vf, af

    def _build_filter_args(
        self, vf: str, af: str
    ) -> Tuple[List[str], List[str], List[Path]]:
        if len(vf) + len(af) < 8000:
            return ["-vf", vf], ["-af", af], []

        script_paths: List[Path] = []
        vf_script = self._write_filter_script(vf, "vf")
        af_script = self._write_filter_script(af, "af")
        script_paths.extend([vf_script, af_script])
        return (
            ["-filter_script:v", str(vf_script)],
            ["-filter_script:a", str(af_script)],
            script_paths,
        )

    @staticmethod
    def _write_filter_script(content: str, prefix: str) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{prefix}.ffscript", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            return Path(tmp.name)

    def _validate_roi(self, roi: Mapping[str, object]) -> bool:
        if not isinstance(roi, dict):
            return False

        normalized = normalize_roi_keys(roi)
        if not normalized:
            return False

        roi = cast(dict[str, int], normalized)

        try:
            roi_box = VideoRoiBox.model_validate(roi)

            if any(
                val < 0 for val in [roi_box.x, roi_box.y, roi_box.width, roi_box.height]
            ):
                logger.warning(f"ROI contains negative values: {roi}")
                return False

            if roi_box.width == 0 or roi_box.height == 0:
                logger.warning(f"ROI has zero width or height: {roi}")
                return False

            if any(
                val > 5000
                for val in [roi_box.x, roi_box.y, roi_box.width, roi_box.height]
            ):
                logger.warning(f"ROI values seem unreasonably large: {roi}")
                return False

            return True

        except (TypeError, ValueError) as e:
            logger.warning(f"ROI contains invalid values: {roi}, error: {e}")
            return False
