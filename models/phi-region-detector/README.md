---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - medical-imaging
  - anonymization
  - phi-detection
  - yolov8
  - onnx
---

# LX Anonymizer PHI region detector

This is the recall-oriented YOLOv8n region detector used as an additive
safeguard by [lx-anonymizer](https://github.com/wg-lux/lx-anonymizer). It
localizes text-like regions that may contain burned-in protected health
information (PHI). Its detections are combined with the pipeline's OCR and EAST
detectors; this model is not a complete anonymization system.

## Files

- `model.onnx`: deployment artifact for lx-anonymizer and ONNX Runtime.
- `model.pt`: Ultralytics checkpoint for evaluation or further fine-tuning.
- `config.json`: preprocessing and output contract.
- `training_metrics.json`: validation metrics and artifact checksums.
- `training_args.yaml`: reproducibility settings without workstation-specific
  paths.

## Inference contract

The model accepts RGB images letterboxed to 960 x 960. It predicts one class,
`phi`. The ONNX output uses YOLO `xywh` boxes followed by class scores. The
lx-anonymizer defaults are a confidence threshold of `0.05` and an NMS IoU
threshold of `0.45`; select the confidence threshold on representative local
validation data when recall requirements differ.

Set the downloaded ONNX path in the application environment:

```bash
PHI_REGION_DETECTOR_MODEL_PATH=/absolute/path/to/model.onnx
PHI_REGION_DETECTOR_MODEL_SHA256=61c2b58e283733c391c577df36dce057d7424f5d611d27ae0867e30ff684a5bd
PHI_REGION_DETECTOR_CONFIDENCE=0.05
PHI_REGION_DETECTOR_INPUT_SIZE=960
PHI_REGION_DETECTOR_RESIZE_MODE=letterbox
```

Ultralytics usage of the PyTorch checkpoint:

```python
from ultralytics import YOLO

model = YOLO("model.pt")
results = model.predict("image.png", imgsz=960, conf=0.05)
```

## Training and evaluation

The checkpoint was refined on a patient-disjoint development split assembled
from RadPHI, synthetic medical-image overlays, synthetic endoscopy stickers,
and MIDI-B validation material. The official MIDI-B test collection was not
used. Training used seed 0, AdamW, a 960 px input, and a single `phi` class.

The internal validation curve reached mAP50 `0.88879` (epoch 39) and
mAP50-95 `0.67799` (epoch 43). These are training-run validation figures, not
official MIDI-B benchmark results. The small positive validation set makes the
estimates uncertain and unsuitable for cross-model claims.

## Intended use and limitations

Use this detector only as one layer in a defense-in-depth anonymization
pipeline. It does not identify DICOM metadata PHI, guarantee that every visible
identifier is found, or determine whether localized text is identifying. Image
styles, modalities, manufacturers, overlays, and languages outside the training
distribution may reduce recall. Validate false negatives per modality and
manufacturer before clinical or production use, and retain human review where
the risk assessment requires it.

The weights derive from Ultralytics YOLO models. Review the AGPL-3.0 obligations
and Ultralytics' alternative licensing terms before distribution or deployment.
