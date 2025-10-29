from typing import Any, Dict, Optional

def normalize_roi_keys(roi: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """
    Normalize ROI key naming conventions.

    Converts between EndoscopyProcessor-style fields
    (e.g., endoscope_image_x) and standard 'x', 'y', 'width', 'height' form.

    Args:
        roi: ROI dict with mixed or inconsistent keys.

    Returns:
        Normalized ROI dict {x, y, width, height} or None if invalid.
    """
    if not roi or not isinstance(roi, dict):
        return None

    key_map = {
        "image_width": roi.get("image_width"),
        "image_height": roi.get("image_height"),
        "x": roi.get("x") or roi.get("endoscope_image_x") or roi.get("endoscope_type_x") or roi.get("patient_first_name_x"),
        "y": roi.get("y") or roi.get("endoscope_image_y") or roi.get("endoscope_type_y") or roi.get("patient_first_name_y"),
        "width": roi.get("width") or roi.get("endoscope_image_width") or roi.get("endoscope_type_width") or roi.get("patient_first_name_width"),
        "height": roi.get("height") or roi.get("endoscope_image_height") or roi.get("endoscope_type_height") or roi.get("patient_first_name_height"),
    }

    # Remove None and negative entries
    if any(v is None or v < 0 for v in key_map.values()):
        return None

    return key_map
