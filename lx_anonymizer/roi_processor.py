from typing import TypedDict, NotRequired, Any, Optional
from functools import reduce

class ROI(TypedDict):
    x: int
    y: int
    width: int
    height: int
    label: NotRequired[str]


class ROIProcessor:
    """Retrieve ROI regions from nested processor dictionaries."""
    
    def __init__(self, processor_name: Optional[str] = None):
        self.processor_name = processor_name

    def deep_get_roi_param(self, data: dict[str, Any], keys: list[str], default: int = 0) -> int:
        """Retrieve a numeric value from a nested dictionary using a list of keys."""
        result = reduce(
            lambda acc, key: acc.get(key, {}) if isinstance(acc, dict) else {},
            keys,
            data
        )
        return result if isinstance(result, int) else default

    def get_roi(self, data: dict[str, Any], roi_key: str, processor_name: Optional[str] = None) -> ROI:
        """
        Retrieve the full ROI {x,y,width,height} for a given ROI key (e.g. 'name_roi').
        
        Example:
            get_roi(data, 'name_roi', 'olympus_cv_1500')
        """
        proc = processor_name or self.processor_name
        if not proc:
            raise ValueError("Processor name not set.")

        roi_dict = data.get(proc, {}).get(roi_key, {})
        x = int(roi_dict.get("x", 0))
        y = int(roi_dict.get("y", 0))
        width = int(roi_dict.get("width", roi_dict.get("w", 0)))
        height = int(roi_dict.get("height", roi_dict.get("h", 0)))
        return {"x": x, "y": y, "width": width, "height": height}
