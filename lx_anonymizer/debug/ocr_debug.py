import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Tuple, Optional

def visualize_ocr_regions(
    frame: np.ndarray,
    regions: List[Tuple[int, int, int, int]],
    texts: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    title: str = "OCR Region Debug View"
) -> None:
    """
    Visualize detected text regions with OCR output overlay.
    
    Args:
        frame: grayscale or RGB numpy array
        regions: list of (x, y, w, h)
        texts: optional list of OCR text strings per region
        confidences: optional list of confidence values (0.0–1.0)
        title: window title for matplotlib
    """
    # Convert to RGB if grayscale
    if frame.ndim == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        vis = frame.copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(vis, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

    for i, (x, y, w, h) in enumerate(regions):
        color = (0, 255, 0) if confidences is None or confidences[i] >= 0.4 else (255, 0, 0)
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=np.array(color)/255.0, facecolor='none')
        ax.add_patch(rect)

        label = ""
        if texts and i < len(texts):
            label = texts[i][:30].replace("\n", " ")
        if confidences and i < len(confidences):
            label += f" ({confidences[i]*100:.0f}%)"
        if label:
            ax.text(x, y - 3, label, color='yellow', fontsize=7,
                    bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none'))

    plt.tight_layout()
    plt.show()
