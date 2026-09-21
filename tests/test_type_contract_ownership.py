"""Prevent per-module copies of the shared runtime contracts."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "names",
    [
        {"Box", "PhiRegion", "ImagePosition", "Roi"},
        {"ImageArray", "FrameArray", "GrayArray", "PixelArray"},
        {"ModifiedImageMap", "OcrResult"},
        {"OcrConfig", "NamedOcrConfig", "OCRConfig", "PytesseractConfig"},
        {"DnnNet", "Cv2DnnModule", "_DnnNet", "_Cv2DnnModule"},
        {"PydicomReader", "_PydicomModule"},
        {"Cv2ImageFunc", "PsmValue", "SplitName"},
        {
            "_GenderDetector",
            "_TesseractModule",
            "TesseractModule",
            "_PytesseractModule",
            "_TesseractOutput",
            "TesseractOutput",
            "TesseractData",
            "PytesseractData",
            "_TesseractData",
        },
    ],
)
def test_runtime_contracts_have_one_owner(names: set[str]) -> None:
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for directory in (root / "lx_anonymizer", root / "tests"):
        for path in directory.rglob("*"):
            if path.suffix not in {".py", ".pyi"} or path.name == "runtime_types.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                elif isinstance(node, ast.TypeAlias):
                    targets = [node.name]
                elif isinstance(node, ast.ClassDef):
                    if node.name in names:
                        violations.append(f"{path.relative_to(root)}:{node.lineno}")
                    continue
                else:
                    continue
                if any(
                    isinstance(target, ast.Name) and target.id in names
                    for target in targets
                ):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, "Import the shared contract instead: " + ", ".join(
        violations
    )


def test_typed_dict_fields_are_not_copied_under_new_names() -> None:
    """Renaming a local helper must not evade the shared-contract ownership rule."""
    root = Path(__file__).resolve().parents[1] / "lx_anonymizer"
    owners = {
        frozenset(
            {("lang", "str"), ("psm", "int"), ("oem", "int"), ("dpi", "int")}
        ): root / "runtime_types.py",
        frozenset(
            {("x", "int"), ("y", "int"), ("width", "int"), ("height", "int")}
        ): root / "utils" / "roi_normalization.py",
    }
    violations: list[str] = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(
                isinstance(base, ast.Name) and base.id == "TypedDict"
                for base in node.bases
            ):
                continue
            # Partial input dictionaries intentionally differ from required internal shapes.
            if any(
                keyword.arg == "total"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in node.keywords
            ):
                continue
            fields = frozenset(
                (field.target.id, ast.unparse(field.annotation))
                for field in node.body
                if isinstance(field, ast.AnnAssign)
                and isinstance(field.target, ast.Name)
            )
            for signature, owner in owners.items():
                if signature <= fields and path != owner:
                    violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, "Inherit or import the shared contract: " + ", ".join(
        violations
    )


def test_adaptive_threshold_has_one_external_boundary() -> None:
    root = Path(__file__).resolve().parents[1] / "lx_anonymizer"
    accesses = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Attribute) and node.attr == "adaptiveThreshold"
    ]
    assert accesses == ["ocr/ocr_preprocessing.py"]
