import logging

from lx_anonymizer.utils.roi_normalization import normalize_roi_keys


def test_normalize_roi_keys_accepts_aliases_and_int_like_values() -> None:
    roi = {
        "endoscope_image_x": "10",
        "endoscope_image_y": 20.0,
        "endoscope_image_width": 300,
        "endoscope_image_height": "400",
    }
    assert normalize_roi_keys(roi) == {"x": 10, "y": 20, "width": 300, "height": 400}


def test_normalize_roi_keys_preserves_zero_values() -> None:
    roi = {"x": 0, "y": 0, "width": 10, "height": 20}
    assert normalize_roi_keys(roi) == {"x": 0, "y": 0, "width": 10, "height": 20}


def test_normalize_roi_keys_returns_none_for_missing_keys(caplog) -> None:
    caplog.set_level(logging.WARNING)
    roi = {"x": 1, "y": 2, "width": 3}
    assert normalize_roi_keys(roi) is None
    assert "missing required ROI keys" in caplog.text


def test_normalize_roi_keys_returns_none_for_invalid_type(caplog) -> None:
    caplog.set_level(logging.WARNING)
    roi = {"x": 1, "y": 2, "width": ["bad"], "height": 4}
    assert normalize_roi_keys(roi) is None
    assert "failed validation" in caplog.text


def test_normalize_roi_keys_returns_none_for_negative_values(caplog) -> None:
    caplog.set_level(logging.WARNING)
    roi = {"x": -1, "y": 2, "width": 3, "height": 4}
    assert normalize_roi_keys(roi) is None
    assert "failed validation" in caplog.text
