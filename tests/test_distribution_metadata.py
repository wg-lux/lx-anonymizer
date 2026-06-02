import ast
import re
import tomllib
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


def _dependency_name(dependency: str) -> str:
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)", dependency)
    assert match is not None
    return match.group(1).lower().replace("_", "-")


def _pyproject() -> dict[str, object]:
    return tomllib.loads(Path("pyproject.toml").read_text())


def test_base_runtime_dependencies_are_release_clean() -> None:
    project = _pyproject()["project"]
    dependencies = {
        _dependency_name(dependency)
        for dependency in project["dependencies"]  # type: ignore[index]
    }

    assert dependencies.isdisjoint(FORBIDDEN_BASE_REQUIREMENTS)


def test_optional_extras_own_tooling_and_accelerators() -> None:
    optional = _pyproject()["project"][  # type: ignore[index]
        "optional-dependencies"
    ]

    dev = {
        _dependency_name(dependency)
        for dependency in optional["dev"]  # type: ignore[index]
    }
    ocr = {
        _dependency_name(dependency)
        for dependency in optional["ocr"]  # type: ignore[index]
    }
    training = {
        _dependency_name(dependency)
        for dependency in optional["training"]  # type: ignore[index]
    }

    assert {"pre-commit", "types-requests", "ziglang"} <= dev
    assert {"onnxruntime", "rapidocr", "tesserocr"} <= ocr
    assert "ultralytics" in training


def test_maturin_excludes_local_artifacts() -> None:
    maturin = _pyproject()["tool"]["maturin"]  # type: ignore[index]
    exclude = set(maturin["exclude"])  # type: ignore[index]

    expected_patterns = {
        ".gitignore",
        "**/__pycache__/**/*",
        "**/*.py[cod]",
        "lx_anonymizer/.gitignore",
        "lx_anonymizer/**/*.bak",
        "lx_anonymizer/generated_reports/**/*",
        "lx_anonymizer/test_images/**/*",
        "lx_anonymizer/unused_scripts/**/*",
        "study_data/**/*",
        "study-data/**/*",
    }

    assert expected_patterns <= exclude


def test_frame_cleaner_keeps_video_mixin_wired() -> None:
    source = Path("lx_anonymizer/frame_cleaner.py").read_text()
    tree = ast.parse(source)

    imports_video_mixin = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "lx_anonymizer.frame_cleaner_video"
        and any(alias.name == "FrameCleanerVideoMixin" for alias in node.names)
        for node in tree.body
    )
    assert imports_video_mixin

    frame_cleaner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FrameCleaner"
    )
    assert any(
        isinstance(base, ast.Name) and base.id == "FrameCleanerVideoMixin"
        for base in frame_cleaner_class.bases
    )
