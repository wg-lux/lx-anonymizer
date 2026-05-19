from pathlib import Path

from lx_anonymizer.anonymization import masking as masking_module
from lx_anonymizer.anonymization.masking import MaskApplication, MaskMode
from lx_anonymizer.video_processing.video_encoder import VideoEncoder


def test_create_mask_config_from_roi_falls_back_to_loaded_dimensions(monkeypatch):
    monkeypatch.setattr(VideoEncoder, "_detect_nvenc_support", lambda self: False)

    mask_app = MaskApplication(preferred_encoder={"type": "cpu"})
    mask_app.default_mask_config = {"image_width": 2048, "image_height": 1152}

    mask_config = mask_app.create_mask_config_from_roi(
        {"x": 550, "y": 0, "width": 1350, "height": 1080}
    )

    assert mask_config["image_width"] == 2048
    assert mask_config["image_height"] == 1152
    assert mask_config["x"] == 550
    assert mask_config["y"] == 0
    assert mask_config["width"] == 1350
    assert mask_config["height"] == 1080
    assert "endoscope_image_x" not in mask_config
    assert "endoscope_image_y" not in mask_config
    assert "endoscope_image_width" not in mask_config
    assert "endoscope_image_height" not in mask_config

    explicit_mask_config = mask_app.create_mask_config_from_roi(
        {
            "x": 550,
            "y": 0,
            "width": 1350,
            "height": 1080,
            "image_width": 1920,
            "image_height": 1080,
        }
    )

    assert explicit_mask_config["image_width"] == 1920
    assert explicit_mask_config["image_height"] == 1080


def test_mask_video_streaming_preserves_dimensions_by_default(monkeypatch, tmp_path):
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
        if video_path == input_video:
            return {"width": 1920, "height": 1080}
        return {"width": 1920, "height": 1080}

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
    assert captured["cmd"][vf_idx] == (
        "drawbox=0:0:551:ih:color=black@1:t=fill,drawbox=0:0:iw:1:color=black@1:t=fill"
    )
    assert calls == [input_video, output_video]


def test_mask_video_streaming_crop_mode_keeps_legacy_behavior(monkeypatch, tmp_path):
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

    def fake_detect(video_path):
        if video_path == input_video:
            return {"width": 1920, "height": 1080}
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

    ok = mask_app.mask_video_streaming(
        input_video,
        mask_config,
        output_video,
        mode=MaskMode.CROP,
    )

    assert ok is True
    vf_idx = captured["cmd"].index("-vf") + 1
    assert captured["cmd"][vf_idx] == "crop=1368:1078:552:2"


def test_mask_video_streaming_scales_mask_for_small_input(monkeypatch, tmp_path):
    monkeypatch.setattr(VideoEncoder, "_detect_nvenc_support", lambda self: False)
    mask_app = MaskApplication(preferred_encoder={"type": "cpu"})
    monkeypatch.setattr(mask_app, "build_encoder_cmd", lambda *_args, **_kwargs: [])

    input_video = tmp_path / "input_small.mp4"
    output_video = tmp_path / "output_small.mp4"
    input_video.write_bytes(b"0" * 100)

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"1" * 50)

        class Result:
            stderr = ""

        return Result()

    def fake_detect(video_path):
        if video_path == input_video:
            return {"width": 480, "height": 270}
        return {"width": 480, "height": 270}

    monkeypatch.setattr(masking_module.subprocess, "run", fake_run)
    monkeypatch.setattr(masking_module.video_utils, "detect_video_format", fake_detect)

    mask_config = {
        "image_width": 1920,
        "image_height": 1080,
        "endoscope_image_x": 550,
        "endoscope_image_y": 0,
        "endoscope_image_width": 1350,
        "endoscope_image_height": 1080,
    }

    ok = mask_app.mask_video_streaming(input_video, mask_config, output_video)

    assert ok is True
    vf_idx = captured["cmd"].index("-vf") + 1
    assert captured["cmd"][vf_idx] == (
        "drawbox=0:0:138:ih:color=black@1:t=fill,"
        "drawbox=476:0:iw-476:ih:color=black@1:t=fill"
    )


def test_backfill_preserved_dimensions_repairs_cropped_output(monkeypatch, tmp_path):
    monkeypatch.setattr(VideoEncoder, "_detect_nvenc_support", lambda self: False)
    mask_app = MaskApplication(preferred_encoder={"type": "cpu"})
    monkeypatch.setattr(mask_app, "build_encoder_cmd", lambda *_args, **_kwargs: [])

    source_video = tmp_path / "source.mp4"
    anonymized_video = tmp_path / "anonymized.mp4"
    source_video.write_bytes(b"source-video")
    anonymized_video.write_bytes(b"cropped-video")

    def fake_run(cmd, *args, **kwargs):
        Path(cmd[-1]).write_bytes(b"repaired-video")

        class Result:
            stderr = ""

        return Result()

    def fake_detect(video_path):
        if video_path == source_video:
            return {"width": 1920, "height": 1080}
        if video_path == anonymized_video:
            return {"width": 1350, "height": 1080}
        return {"width": 1920, "height": 1080}

    monkeypatch.setattr(masking_module.subprocess, "run", fake_run)
    monkeypatch.setattr(masking_module.video_utils, "detect_video_format", fake_detect)

    result = mask_app.backfill_preserved_dimensions(
        source_video=source_video,
        anonymized_video=anonymized_video,
        mask_config={
            "image_width": 1920,
            "image_height": 1080,
            "endoscope_image_x": 550,
            "endoscope_image_y": 0,
            "endoscope_image_width": 1350,
            "endoscope_image_height": 1080,
        },
    )

    assert result.status == "repaired"
    assert result.repaired is True
    assert anonymized_video.read_bytes() == b"repaired-video"


def test_backfill_preserved_dimensions_dry_run_does_not_replace(monkeypatch, tmp_path):
    monkeypatch.setattr(VideoEncoder, "_detect_nvenc_support", lambda self: False)
    mask_app = MaskApplication(preferred_encoder={"type": "cpu"})

    source_video = tmp_path / "source.mp4"
    anonymized_video = tmp_path / "anonymized.mp4"
    source_video.write_bytes(b"source-video")
    anonymized_video.write_bytes(b"cropped-video")

    def fake_detect(video_path):
        if video_path == source_video:
            return {"width": 1920, "height": 1080}
        return {"width": 1350, "height": 1080}

    monkeypatch.setattr(masking_module.video_utils, "detect_video_format", fake_detect)

    result = mask_app.backfill_preserved_dimensions(
        source_video=source_video,
        anonymized_video=anonymized_video,
        mask_config={},
        dry_run=True,
    )

    assert result.status == "would_repair"
    assert result.repaired is False
    assert anonymized_video.read_bytes() == b"cropped-video"
