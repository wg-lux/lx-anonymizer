import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType


def _load_audit_module() -> ModuleType:
    module_path = Path(__file__).parents[1] / "scripts" / "audit_distribution.py"
    spec = importlib.util.spec_from_file_location("audit_distribution", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wheel(
    path: Path,
    members: dict[str, str],
    metadata: str | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        archive.writestr(
            "lx_anonymizer-0.0.0.dist-info/METADATA",
            metadata or "Metadata-Version: 2.4\nName: lx-anonymizer\nVersion: 0.0.0\n",
        )
        archive.writestr(
            "lx_anonymizer/_lx_anonymizer_native.cpython-312-x86_64-linux-gnu.so",
            "",
        )


def test_distribution_audit_rejects_unwired_frame_cleaner_video_mixin(
    tmp_path: Path,
) -> None:
    audit_distribution = _load_audit_module()
    wheel = tmp_path / "lx_anonymizer-0.0.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        {
            "lx_anonymizer/frame_cleaner.py": """
class FrameCleaner:
    def _analyze_video_frames(self, video_path, total_frames):
        for idx, frame, stride in self._iter_video(video_path, total_frames):
            pass
""",
            "lx_anonymizer/frame_cleaner_video.py": """
class FrameCleanerVideoMixin:
    def _iter_video(self, video_path, total_frames):
        yield from ()
""",
        },
    )

    errors = audit_distribution.audit_artifact(wheel)

    assert any(
        "neither defines it nor inherits FrameCleanerVideoMixin" in e for e in errors
    )


def test_distribution_audit_accepts_wired_frame_cleaner_video_mixin(
    tmp_path: Path,
) -> None:
    audit_distribution = _load_audit_module()
    wheel = tmp_path / "lx_anonymizer-0.0.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        {
            "lx_anonymizer/frame_cleaner.py": """
from lx_anonymizer.frame_cleaner_video import FrameCleanerVideoMixin


class FrameCleaner(FrameCleanerVideoMixin):
    def _analyze_video_frames(self, video_path, total_frames):
        for idx, frame, stride in self._iter_video(video_path, total_frames):
            pass
""",
            "lx_anonymizer/frame_cleaner_video.py": """
class FrameCleanerVideoMixin:
    def _iter_video(self, video_path, total_frames):
        yield from ()
""",
        },
    )

    assert audit_distribution.audit_artifact(wheel) == []


def test_distribution_audit_allows_tesserocr_base_dependency(tmp_path: Path) -> None:
    audit_distribution = _load_audit_module()
    wheel = tmp_path / "lx_anonymizer-0.0.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        {
            "lx_anonymizer/frame_cleaner.py": """
class FrameCleaner:
    pass
""",
        },
        metadata=(
            "Metadata-Version: 2.4\n"
            "Name: lx-anonymizer\n"
            "Version: 0.0.0\n"
            "Requires-Dist: tesserocr>=2.9.1\n"
        ),
    )

    assert audit_distribution.audit_artifact(wheel) == []
