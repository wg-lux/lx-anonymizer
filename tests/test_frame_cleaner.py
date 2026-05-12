import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import numpy.typing as npt

# Adjust import based on your actual package structure
from lx_anonymizer.frame_cleaner import FrameCleaner
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


def _frame_cleaner_unit_stub() -> FrameCleaner:
    frame_cleaner = FrameCleaner.__new__(FrameCleaner)
    frame_cleaner.frame_ocr = MagicMock()
    frame_cleaner.frame_metadata_extractor = FrameMetadataExtractor()
    frame_cleaner.sensitive_meta = SensitiveMeta()
    frame_cleaner.frame_collection = []
    frame_cleaner.frame_observations = []
    frame_cleaner.ocr_text_collection = []
    return frame_cleaner


class TestFrameCleanerRefactored:
    @pytest.fixture
    def frame_cleaner(self) -> FrameCleaner:
        """Fixture for an initialized FrameCleaner."""
        # Initialize with LLM disabled to speed up tests and avoid API calls
        return FrameCleaner(use_llm=False)

    @pytest.fixture
    def mock_frame(self):
        """Create a dummy grayscale numpy frame (height, width)."""
        return np.zeros((1080, 1920), dtype=np.uint8)

    def test_iter_video_logic(
        self,
        frame_cleaner: FrameCleaner,
    ):
        """
        Test _iter_video generator logic without a real file.

        This verifies:
        1. It opens the video.
        2. It calculates the skip rate.
        3. It yields frames at the correct intervals.
        4. It handles the end of the stream correctly.
        """
        total_frames = 100
        video_path = Path("dummy_video.mp4")

        # Setup the Mock VideoCapture
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap_cls.return_value = mock_cap

            # Setup Metadata
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                # cv2.CAP_PROP_FPS is usually 5
                5: 30.0
            }.get(prop, 0)

            # Setup the stream: Return a frame 100 times, then False (EOF)
            # side_effect: [(True, frame), (True, frame), ... (False, None)]
            # We create a dummy color frame (H, W, 3) because _iter_video converts BGR2GRAY
            dummy_color_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            # Create a side effect that yields frames and then stops
            read_side_effect = [(True, dummy_color_frame)] * total_frames + [
                (False, None)
            ]
            mock_cap.read.side_effect = read_side_effect

            # Execute the generator
            # Pass total_frames manually as clean_video usually calculates this before calling _iter_video
            iterator = frame_cleaner._iter_video(video_path, total_frames=total_frames)

            yielded_items = list(iterator)

            # Assertions
            mock_cap_cls.assert_called_with(str(video_path))
            assert mock_cap.release.called

            # Validate output structure: (index, frame, stride)
            assert len(yielded_items) > 0
            idx, frame, stride = yielded_items[0]

            assert isinstance(idx, int)
            assert isinstance(frame, np.ndarray)
            assert isinstance(stride, int)
            assert frame.ndim == 2  # Ensure it was converted to grayscale

            # Validate sampling logic using the current helper behavior.
            target_samples = frame_cleaner._target_sample_count(total_frames)
            expected_skip = max(5, min(-(-total_frames // target_samples), 60))
            assert stride == expected_skip

            # Check indices: 0, 5, 10, ...
            indices = [item[0] for item in yielded_items]
            assert indices[1] - indices[0] == expected_skip

    def test_clean_video_pipeline_integration(
        self,
        frame_cleaner: FrameCleaner,
        mock_frame: npt.NDArray[np.uint8],
    ) -> None:
        """
        Test clean_video by mocking _iter_video.

        This avoids the complexity of cv2 mocks and focuses on the
        FrameCleaner pipeline:
        Iteration -> Processing -> Metadata Merge -> Decision -> Output
        """
        video_path = Path("input.mp4")
        output_path = Path("output.mp4")

        # Define 3 frames to be yielded by the generator
        # Frame 0: Sensitive
        # Frame 1: Not Sensitive
        # Frame 2: Sensitive

        # (index, frame, stride)
        mock_stream_data = [(0, mock_frame, 1), (1, mock_frame, 1), (2, mock_frame, 1)]

        # Patch internal dependencies
        with (
            patch.object(
                frame_cleaner, "_iter_video", return_value=mock_stream_data
            ) as mock_iter,
            patch.object(frame_cleaner, "_process_frame_single") as mock_process,
            patch.object(
                frame_cleaner, "remove_frames_from_video_streaming"
            ) as mock_remove,
            patch("cv2.VideoCapture") as mock_cv2,
        ):  # Mock cap just for frame count check
            # Setup Metadata extraction simulation
            # _process_frame_single returns: (is_sensitive, frame_meta, ocr_text, ocr_conf)
            mock_process.side_effect = [
                (True, {"patient_last_name": "Smith"}, "Smith", 0.9),
                (False, {}, "Clean", 0.9),
                (True, {"patient_last_name": "Doe"}, "Doe", 0.9),
            ]

            # Mock the total frame count check at start of clean_video
            mock_cv2_instance = MagicMock()
            mock_cv2.return_value = mock_cv2_instance
            mock_cv2_instance.get.return_value = 100.0

            # EXECUTE
            result_path, meta = frame_cleaner.clean_video(
                video_path=video_path,
                endoscope_image_roi=None,
                endoscope_data_roi_nested=None,
                output_path=output_path,
                technique="remove_frames",
            )

            # ASSERTIONS

            # 1. Verify Iteration
            mock_iter.assert_called_once()

            # 2. Verify Processing calls
            assert mock_process.call_count == 3

            # 3. Verify Removal Logic
            # Should have accumulated indices [0, 2] as sensitive
            mock_remove.assert_called_once()
            args, _ = mock_remove.call_args
            # args[1] is sensitive_idx
            assert args[1] == [0, 2]

            # 4. Verify Metadata Persistence
            assert "patient_first_name" in meta
            # Fill-only semantics preserve the first nonblank value.
            assert meta["patient_last_name"] == "Smith"

    def test_process_frame_marks_sensitive_after_ocr_metadata_merge(
        self,
        mock_frame: npt.NDArray[np.uint8],
    ) -> None:
        frame_cleaner = _frame_cleaner_unit_stub()
        ocr_text = "Patient: Thomas Lux geb. 15.02.2024"
        ocr_meta = {"backend": "test", "words": 5}
        parsed_meta = {
            "patient_first_name": "Thomas",
            "patient_last_name": "Lux",
            "patient_dob": "2024-02-15",
        }

        with (
            patch.object(
                frame_cleaner.frame_ocr,
                "extract_text_from_frame",
                return_value=(ocr_text, 0.92, ocr_meta),
            ),
            patch.object(
                frame_cleaner,
                "_unified_metadata_extract",
                return_value=parsed_meta,
            ),
        ):
            is_sensitive, frame_meta, returned_text, returned_conf = (
                frame_cleaner._process_frame_single(
                    mock_frame,
                    endoscope_image_roi=None,
                    endoscope_data_roi_nested=None,
                    frame_id=7,
                    collect_for_batch=True,
                )
            )

        assert is_sensitive is True
        assert frame_meta["patient_first_name"] == "Thomas"
        assert frame_meta["patient_last_name"] == "Lux"
        assert frame_meta["patient_dob"] == "2024-02-15"
        assert returned_text == ocr_text
        assert returned_conf == 0.92
        assert frame_cleaner.frame_collection[0]["is_sensitive"] is True
        observation = frame_cleaner.frame_observations[0]
        assert observation["frame_number"] == 7
        assert observation["metadata_signal"] is True
        assert observation["is_sensitive"] is True
        assert "metadata_signal" in observation["source_tags"]

    def test_process_frame_uses_image_roi_as_ocr_roi_when_nested_roi_missing(
        self,
        mock_frame: npt.NDArray[np.uint8],
    ) -> None:
        frame_cleaner = _frame_cleaner_unit_stub()
        image_roi = {
            "endoscope_image_x": "10",
            "endoscope_image_y": 20.0,
            "endoscope_image_width": 300,
            "endoscope_image_height": 120,
        }

        with patch.object(
            frame_cleaner.frame_ocr,
            "extract_text_from_frame",
            return_value=("", 0.0, {}),
        ) as mock_extract:
            frame_cleaner._process_frame_single(
                mock_frame,
                endoscope_image_roi=image_roi,
                endoscope_data_roi_nested=None,
                frame_id=7,
            )

        mock_extract.assert_called_once_with(
            mock_frame,
            {"endoscope_image": {"x": 10, "y": 20, "width": 300, "height": 120}},
        )

    def test_process_frame_prefers_explicit_nested_ocr_roi(
        self,
        mock_frame: npt.NDArray[np.uint8],
    ) -> None:
        frame_cleaner = _frame_cleaner_unit_stub()
        image_roi = {"x": 10, "y": 20, "width": 300, "height": 120}
        nested_roi = {"patient_info": {"x": 1, "y": 2, "width": 3, "height": 4}}

        with patch.object(
            frame_cleaner.frame_ocr,
            "extract_text_from_frame",
            return_value=("", 0.0, {}),
        ) as mock_extract:
            frame_cleaner._process_frame_single(
                mock_frame,
                endoscope_image_roi=image_roi,
                endoscope_data_roi_nested=nested_roi,
                frame_id=7,
            )

        mock_extract.assert_called_once_with(mock_frame, nested_roi)

    def test_process_frame_records_phi_detector_observations_as_sensitive(
        self,
        mock_frame: npt.NDArray[np.uint8],
    ) -> None:
        frame_cleaner = _frame_cleaner_unit_stub()
        phi_regions = [
            {
                "source": "phi_detector",
                "x": 30,
                "y": 40,
                "width": 50,
                "height": 60,
                "x1": 30,
                "y1": 40,
                "x2": 80,
                "y2": 100,
                "confidence": None,
                "class_id": None,
            }
        ]

        with (
            patch.object(
                frame_cleaner.frame_ocr,
                "extract_text_from_frame",
                return_value=("", 0.0, {}),
            ),
            patch.object(
                frame_cleaner,
                "_detect_phi_regions_for_frame",
                return_value=phi_regions,
            ),
        ):
            is_sensitive, frame_meta, returned_text, returned_conf = (
                frame_cleaner._process_frame_single(
                    mock_frame,
                    endoscope_image_roi=None,
                    endoscope_data_roi_nested=None,
                    frame_id=11,
                    collect_for_batch=True,
                )
            )

        assert is_sensitive is True
        assert frame_meta["patient_first_name"] is None
        assert returned_text == ""
        assert returned_conf == 0.0
        observation = frame_cleaner.frame_observations[0]
        assert observation["frame_number"] == 11
        assert observation["phi_regions"] == phi_regions
        assert "phi_detector" in observation["source_tags"]
        assert frame_cleaner.frame_collection == []

    def test_sampling_logic_long_video(self, frame_cleaner: FrameCleaner) -> None:
        """
        Specifically test that long videos trigger higher skip rates
        to optimize memory/cpu usage without loading actual frames.
        """
        video_path = Path("long.mp4")
        long_frame_count = 50000

        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap_cls.return_value = mock_cap

            # FPS = 30
            mock_cap.get.side_effect = lambda x: 30.0 if x == 5 else 0

            # Mock read to stop immediately so we don't loop 50k times
            # We only care about the setup logic before the loop yields
            mock_cap.read.return_value = (False, None)

            # We iterate manually to inspect the logic inside _iter_video
            # Since we can't easily inspect local variables in a generator,
            # we check the stride of the *first* yielded item if we allowed one read.

            # Let's allow one read to capture the calculated stride
            dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
            mock_cap.read.side_effect = [(True, dummy_frame), (False, None)]

            gen = frame_cleaner._iter_video(video_path, long_frame_count)
            items = list(gen)

            if items:
                _, _, stride = items[0]

                # Logic check:
                # target = 200 samples
                # calc_skip = ceil(50000 / 200) = 250
                # fps = 30
                # skip = max(5, min(250, 30)) -> max(5, 30) -> 30

                assert stride == 60

    def test_clean_video_raises_when_masking_fails(
        self,
        frame_cleaner: FrameCleaner,
        mock_frame: npt.NDArray[np.uint8],
    ) -> None:
        video_path = Path("input.mp4")
        output_path = Path("output.mp4")
        mock_stream_data = [(0, mock_frame, 1)]

        with (
            patch.object(frame_cleaner, "_iter_video", return_value=mock_stream_data),
            patch.object(
                frame_cleaner,
                "_process_frame_single",
                return_value=(False, {}, "Clean", 0.9),
            ),
            patch.object(frame_cleaner, "_mask_video_streaming", return_value=False),
            patch("cv2.VideoCapture") as mock_cv2,
        ):
            mock_cv2_instance = MagicMock()
            mock_cv2.return_value = mock_cv2_instance
            mock_cv2_instance.get.return_value = 10.0

            with pytest.raises(RuntimeError, match="Masking failed"):
                frame_cleaner.clean_video(
                    video_path=video_path,
                    endoscope_image_roi=None,
                    endoscope_data_roi_nested=None,
                    output_path=output_path,
                    technique="mask_overlay",
                )

    def test_ffmpeg_streaming_command_construction(
        self, frame_cleaner: FrameCleaner
    ) -> None:
        """
        Test that the streaming remove function constructs valid FFmpeg commands.
        Uses subprocess patching to avoid actual execution.
        """
        with (
            patch("subprocess.run") as mock_run,
            patch("subprocess.Popen") as mock_popen,
            patch(
                "lx_anonymizer.video_processing.video_utils.detect_video_format"
            ) as mock_fmt,
            patch("lx_anonymizer.video_processing.video_utils.named_pipe") as mock_pipe,
        ):
            # Setup mocks
            mock_fmt.return_value = {"can_stream_copy": True, "has_audio": True}
            mock_pipe.return_value.__enter__.return_value = Path("/tmp/pipe")

            original = Path("in.mp4")
            output = Path("out.mp4")
            remove_indices = [10, 11, 12]

            # Execute
            frame_cleaner.remove_frames_from_video_streaming(
                original, remove_indices, output, total_frames=1000, use_named_pipe=True
            )

            # Verify Popen (Filter process) arguments
            assert mock_popen.called
            filter_args = mock_popen.call_args[0][0]

            # Check for select filter construction
            # vf should contain: select='not(eq(n\,10)+eq(n\,11)+eq(n\,12))'
            filter_string = [arg for arg in filter_args if "select='not(" in arg][0]
            assert "eq(n\\,10)" in filter_string
            assert "eq(n\\,11)" in filter_string

            # Verify Run (Copy process) arguments
            assert mock_run.called
            copy_args = mock_run.call_args[0][0]
            assert "-c" in copy_args
            assert "copy" in copy_args  # Should enable stream copy from pipe
