#!/usr/bin/env python3
"""Audit built Python distribution artifacts for release hygiene."""

from __future__ import annotations

import argparse
import email.parser
import re
import sys
import tarfile
import zipfile
from pathlib import Path


FORBIDDEN_BASE_REQUIREMENTS = {
    "onnxruntime",
    "pre-commit",
    "protobuf",
    "puccinialin",
    "rapidocr",
    "types-requests",
    "ultralytics",
    "ziglang",
}

_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
_NATIVE_EXTENSION_RE = re.compile(
    r"(^|/)lx_anonymizer/_lx_anonymizer_native[^/]*\.so$"
)


def _normalize_requirement_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME_RE.match(requirement)
    if not match:
        return ""
    return match.group(1).lower().replace("_", "-")


def _has_path_segment(path: str, segment: str) -> bool:
    return f"/{segment}/" in f"/{path.strip('/')}/"


def _is_wheel(path: Path) -> bool:
    return path.suffix == ".whl"


def _is_sdist(path: Path) -> bool:
    return path.suffixes[-2:] == [".tar", ".gz"]


def _iter_zip_members(path: Path) -> tuple[list[str], dict[str, str]]:
    metadata: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            if name.endswith(".dist-info/METADATA"):
                metadata[name] = archive.read(name).decode("utf-8")
    return names, metadata


def _iter_tar_members(path: Path) -> tuple[list[str], dict[str, str]]:
    metadata: dict[str, str] = {}
    with tarfile.open(path) as archive:
        names = archive.getnames()
        for member in archive.getmembers():
            if not (member.isfile() and member.name.endswith("/PKG-INFO")):
                continue
            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue
            metadata[member.name] = file_obj.read().decode("utf-8")
    return names, metadata


def _read_artifact_member_by_suffix(path: Path, suffix: str) -> str | None:
    normalized_suffix = suffix.strip("/")

    if _is_wheel(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith(normalized_suffix):
                    return archive.read(name).decode("utf-8")
        return None

    if _is_sdist(path):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if not (member.isfile() and member.name.endswith(normalized_suffix)):
                    continue
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    return None
                return file_obj.read().decode("utf-8")

    return None


def _forbidden_member_reason(name: str, *, wheel: bool) -> str | None:
    normalized = name.replace("\\", "/")

    if _has_path_segment(normalized, "__pycache__"):
        return "contains __pycache__"
    if normalized.endswith((".pyc", ".pyo", ".pyd")):
        return "contains compiled Python cache"
    if normalized.endswith(".bak"):
        return "contains backup file"
    if normalized.endswith(".gitignore"):
        return "contains git metadata"
    if normalized.endswith(".so") and not (
        wheel and _NATIVE_EXTENSION_RE.search(normalized)
    ):
        return "contains a non-maturin shared object"

    forbidden_segments = {
        "generated_reports",
        "study-data",
        "study_data",
        "test_images",
        "unused_scripts",
    }
    for segment in forbidden_segments:
        if _has_path_segment(normalized, segment):
            return f"contains {segment}"

    if wheel and _has_path_segment(normalized, "tests"):
        return "wheel contains tests"

    return None


def _frame_cleaner_consistency_errors(path: Path) -> list[str]:
    frame_cleaner = _read_artifact_member_by_suffix(
        path,
        "lx_anonymizer/frame_cleaner.py",
    )
    if frame_cleaner is None:
        return [f"{path}: missing lx_anonymizer/frame_cleaner.py"]

    if "self._iter_video(" not in frame_cleaner:
        return []

    if re.search(r"^\s+def _iter_video\(", frame_cleaner, re.MULTILINE):
        return []

    inherits_video_mixin = "class FrameCleaner(FrameCleanerVideoMixin)" in frame_cleaner
    if not inherits_video_mixin:
        return [
            f"{path}: FrameCleaner calls _iter_video but neither defines it nor "
            "inherits FrameCleanerVideoMixin"
        ]

    frame_cleaner_video = _read_artifact_member_by_suffix(
        path,
        "lx_anonymizer/frame_cleaner_video.py",
    )
    if frame_cleaner_video is None:
        return [
            f"{path}: FrameCleaner inherits FrameCleanerVideoMixin but "
            "lx_anonymizer/frame_cleaner_video.py is missing"
        ]
    if "def _iter_video(" not in frame_cleaner_video:
        return [
            f"{path}: FrameCleanerVideoMixin is packaged without _iter_video"
        ]

    return []


def _metadata_errors(metadata_text: str, metadata_name: str) -> list[str]:
    errors: list[str] = []
    parsed = email.parser.Parser().parsestr(metadata_text)
    for requirement in parsed.get_all("Requires-Dist", []):
        requirement_lc = requirement.lower()
        if "extra ==" in requirement_lc:
            continue
        requirement_name = _normalize_requirement_name(requirement)
        if requirement_name in FORBIDDEN_BASE_REQUIREMENTS:
            message = (
                f"{metadata_name}: forbidden base dependency "
                f"{requirement_name!r}"
            )
            errors.append(message)
    return errors


def audit_artifact(path: Path) -> list[str]:
    if _is_wheel(path):
        names, metadata = _iter_zip_members(path)
        wheel = True
    elif _is_sdist(path):
        names, metadata = _iter_tar_members(path)
        wheel = False
    else:
        return [f"{path}: unsupported artifact type"]

    errors: list[str] = []
    for name in names:
        reason = _forbidden_member_reason(name, wheel=wheel)
        if reason is not None:
            errors.append(f"{path}: {name}: {reason}")

    errors.extend(_frame_cleaner_consistency_errors(path))

    if wheel and not any(_NATIVE_EXTENSION_RE.search(name) for name in names):
        errors.append(f"{path}: wheel is missing the maturin native extension")

    if not metadata:
        errors.append(f"{path}: package metadata not found")
    for metadata_name, metadata_text in metadata.items():
        errors.extend(_metadata_errors(metadata_text, metadata_name))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit lx-anonymizer wheel and sdist artifacts."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    for artifact in args.artifacts:
        errors.extend(audit_artifact(artifact))

    if errors:
        print("Distribution audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Distribution audit passed for {len(args.artifacts)} artifact(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
