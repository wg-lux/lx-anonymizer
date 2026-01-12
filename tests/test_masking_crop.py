from pathlib import Path

from lx_anonymizer.anonymization import masking as masking_module
from lx_anonymizer.anonymization.masking import MaskApplication
from lx_anonymizer.video_processing.video_encoder import VideoEncoder


def test_mask_video_streaming_crops_and_checks_dimensions(monkeypatch, tmp_path):
    monkeypatch.setattr(VideoEncoder, "_detect_nvenc_support", lambda self: False)

    mask_app = MaskApplication(preferred_encoder={"type": "cpu"})
    monkeypatch.setattr(mask_app, "build_encoder_cmd", lambda *_args, **_kwargs: [])

    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    input_video.write_bytes(b"0" * 100)

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"1" * 50)

        class Result:
            stderr = ""

        return Result()

    calls = []

    def fake_detect(video_path):
        calls.append(video_path)
        return {"width": 1368, "height": 1078}

    monkeypatch.setattr(masking_module.subprocess, "run", fake_run)
    monkeypatch.setattr(masking_module.video_utils, "detect_video_format", fake_detect)

    mask_config = {
        "endoscope_image_x": 551,
        "endoscope_image_y": 1,
        "endoscope_image_width": 1500,
        "endoscope_image_height": 1081,
        "image_width": 1920,
        "image_height": 1080,
    }

    ok = mask_app.mask_video_streaming(input_video, mask_config, output_video)

    assert ok is True
    vf_idx = captured["cmd"].index("-vf") + 1
    assert captured["cmd"][vf_idx] == "crop=1368:1078:552:2"
    assert calls == [output_video]
