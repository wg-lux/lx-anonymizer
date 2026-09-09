# NVIDIA NVENC acceleration

`FrameCleaner` probes FFmpeg for a usable `h264_nvenc` encoder when it is
constructed. When the probe succeeds, video output uses the first GPU with
NVENC; otherwise the pipeline uses `libx264`. This is an encoding optimization,
not a change to the anonymization decision or its safety guarantees.

## Requirements

- An NVIDIA GPU with a driver that supports NVENC.
- An FFmpeg build containing the `h264_nvenc` encoder.
- A runtime where the process can access the GPU.

Check the FFmpeg installation before troubleshooting the application:

```bash
ffmpeg -hide_banner -encoders | grep nvenc
nvidia-smi
```

The first command should list `h264_nvenc`. If it does not, install an FFmpeg
build with NVENC support or use the CPU path.

## Encoder selection

The default selection is automatic:

```python
from lx_anonymizer.frame_cleaner import FrameCleaner

cleaner = FrameCleaner()
result = cleaner.process(request)
```

The selected encoder is reported in the `FrameCleaner` initialization logs. The
current quality profiles are:

| Profile | NVENC | CPU (`libx264`) |
| --- | --- | --- |
| `fast` | `p2`, CQ 25 | `faster`, CRF 20 |
| `balanced` | `p4`, CQ 20 | `veryfast`, CRF 18 |
| `quality` | `p6`, CQ 18 | `slow`, CRF 15 |

Lower CQ/CRF values generally preserve more image quality at the cost of
encoding time and output size. A compatibility fallback uses a faster preset
when the preferred encoder invocation fails.

## Operational behavior

NVENC is used only where the video path needs encoding. Operations that can
retain an existing stream may use stream copy instead. If an NVENC invocation
cannot run, the pipeline may retry with CPU encoding; a failed output must still
be treated as a failed anonymization attempt and validated by the caller.

Every video processing attempt owns its FFmpeg processes, temporary resources,
and output path. See the [video import concurrency contract](VIDEO_IMPORT_CONCURRENCY_CONTRACT.md)
before changing encoder selection, fallback, cancellation, or cleanup behavior.

## Performance guidance

Do not use fixed throughput claims as acceptance criteria. Performance depends
on GPU generation, driver and FFmpeg versions, input resolution, filters, and
concurrent GPU work. Benchmark representative source material and validate the
resulting video before using acceleration in a production workflow.
