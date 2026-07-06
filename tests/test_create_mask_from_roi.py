from lx_anonymizer.frame_cleaner import FrameCleaner


def test_frame_cleaner_initializes_mask_config_from_roi() -> None:
    frame_cleaner = FrameCleaner(use_llm=False)
    test_roi = {
        "x": 550,
        "y": 0,
        "width": 1350,
        "height": 1080,
        "image_width": 1920,
        "image_height": 1080,
    }

    assert frame_cleaner._mask_video_streaming is not None  # pyright: ignore[reportPrivateUsage]
    assert frame_cleaner._create_mask_config_from_roi is not None  # pyright: ignore[reportPrivateUsage]
    mask_config = frame_cleaner._create_mask_config_from_roi(test_roi)  # pyright: ignore[reportPrivateUsage]

    assert mask_config.x == 550
    assert mask_config.width == 1350
