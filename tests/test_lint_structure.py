import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.lint_structure import discover, validate


@pytest.fixture(autouse=True)
def mock_central_video_format() -> None:
    """Context tooling tests do not load the video pipeline fixture."""


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "domain").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "domain" / "api.py").touch()
    (tmp_path / "contract.md").touch()
    write_manifest(tmp_path)
    write_manifest(tmp_path, "domain")
    return tmp_path


def write_manifest(root: Path, directory: str = ".", /, **overrides: object) -> None:
    is_root = directory == "."
    prefix = "" if is_root else "../" * len(Path(directory).parts)
    if not is_root:
        (root / directory / "api.py").touch()
    data: dict[str, object] = {
        "version": 1,
        "tier": "root" if is_root else "domain",
        "path": directory,
        "parent": None if is_root else prefix + "structure.yml",
        "summary": "Fixture context",
        "responsibilities": ["Provide a typed API"],
        "boundaries": ["No storage publication"],
        "entrypoints": []
        if is_root
        else [{"name": "API", "file": "api.py", "purpose": "Compute"}],
        "dependencies": {"internal": ["domain"], "external": ["Python"]},
        "submodules": [
            {
                "name": "domain",
                "path": "domain",
                "manifest": "domain/structure.yml",
                "summary": "Compute",
            }
        ]
        if is_root
        else [],
        "contracts": [{"name": "Contract", "doc": prefix + "contract.md"}],
        "tests": {
            "directory": prefix + "tests",
            "command": "pytest -q",
        },
        "custom_owner": {"name": "Preserved owner"},
    }
    data.update(overrides)
    (root / directory / "structure.yml").write_text(
        yaml.safe_dump(data), encoding="utf-8"
    )


def test_valid_graph_preserves_extras_and_inventory(repository: Path) -> None:
    write_manifest(
        repository, files=["contract.md", {"path": "domain", "owner": "compute"}]
    )
    assert validate(repository) == []


@pytest.mark.parametrize("tracked", [False, True])
def test_discovery_includes_new_and_tracked_manifests(
    repository: Path, tracked: bool
) -> None:
    if tracked:
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
    assert set(discover(repository)) == {
        repository / "structure.yml",
        repository / "domain/structure.yml",
    }


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("version", True, "version"),
        ("version", 2, "version"),
        ("tier", "unknown", "tier"),
        ("summary", "", "summary"),
        ("responsibilities", [], "responsibilities"),
        ("responsibilities", "compute", "responsibilities"),
        ("boundaries", [42], "boundaries"),
        (
            "entrypoints",
            [{"name": "API", "file": "missing.py", "purpose": "Compute"}],
            "entrypoints.file",
        ),
        ("files", ["missing.py"], "files.path"),
        ("files", [{"path": "missing.py"}], "files.path"),
        (
            "dependencies",
            {"internal": ["missing"], "external": []},
            "dependencies.internal",
        ),
        ("dependencies", {"internal": [], "external": [42]}, "dependencies.external"),
        ("contracts", [{"name": "Missing", "doc": "missing.md"}], "contracts.doc"),
        ("tests", {"directory": "missing", "command": "pytest"}, "tests.directory"),
        ("tests", {"directory": "tests", "command": ""}, "tests.command"),
        ("parent", "structure.yml", "root must"),
        ("path", "domain", "does not match"),
        (
            "submodules",
            [
                {
                    "name": "domain",
                    "path": "domain",
                    "manifest": "domain/api.py",
                    "summary": "Compute",
                }
            ],
            "manifest must",
        ),
    ],
)
def test_schema_and_broken_references(
    repository: Path, field: str, value: object, error: str
) -> None:
    write_manifest(repository, **{field: value})
    assert any(error in item for item in validate(repository))


@pytest.mark.parametrize("parent", [None, "../missing.yml", "api.py"])
def test_missing_or_invalid_parent(repository: Path, parent: object) -> None:
    write_manifest(repository, "domain", parent=parent)
    assert any("parent" in item for item in validate(repository))


def test_orphan_child(repository: Path) -> None:
    write_manifest(repository, submodules=[])
    errors = validate(repository)
    assert any("parent does not link back" in item for item in errors)
    assert any("unreachable" in item for item in errors)


def test_parent_must_match_caller(repository: Path) -> None:
    (repository / "domain/nested").mkdir()
    write_manifest(repository, "domain/nested", parent="../../structure.yml")
    write_manifest(
        repository,
        "domain",
        submodules=[
            {
                "name": "nested",
                "path": "nested",
                "manifest": "nested/structure.yml",
                "summary": "Nested",
            }
        ],
    )
    assert any("parent does not match caller" in item for item in validate(repository))


def test_duplicate_child(repository: Path) -> None:
    child = {
        "name": "domain",
        "path": "domain",
        "manifest": "domain/structure.yml",
        "summary": "Compute",
    }
    write_manifest(repository, submodules=[child, child])
    assert any("duplicate child" in item for item in validate(repository))


@pytest.mark.parametrize("reference", ["/etc/passwd", "../outside", "escape"])
def test_escaping_references(repository: Path, reference: str) -> None:
    (repository / "escape").symlink_to(repository.parent, target_is_directory=True)
    write_manifest(repository, files=[reference])
    assert any(
        "must be relative" in item or "escapes repository" in item
        for item in validate(repository)
    )


def test_ignored_manifest_is_not_discovered(repository: Path) -> None:
    (repository / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (repository / ".venv").mkdir()
    (repository / ".venv/structure.yml").write_text("invalid: [", encoding="utf-8")
    assert validate(repository) == []


def test_linked_ignored_manifest_is_rejected(repository: Path) -> None:
    (repository / ".gitignore").write_text("domain/structure.yml\n", encoding="utf-8")
    assert any("ignored by Git" in item for item in validate(repository))


def test_deleted_tracked_manifest(repository: Path) -> None:
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    (repository / "domain/structure.yml").unlink()
    assert any(
        "missing" in item or "No such file" in item for item in validate(repository)
    )


@pytest.mark.parametrize(
    "content", ["summary: [", "- list", "42: value", "summary: only"]
)
def test_malformed_manifest(repository: Path, content: str) -> None:
    (repository / "structure.yml").write_text(content, encoding="utf-8")
    assert validate(repository)


def test_missing_root(repository: Path) -> None:
    (repository / "structure.yml").unlink()
    assert any("missing or invalid root" in item for item in validate(repository))


def test_cli_exit_status(repository: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/lint_structure.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", str(repository)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "valid" in result.stdout
    (repository / "domain/api.py").unlink()
    result = subprocess.run(
        [sys.executable, str(script), "--root", str(repository)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "entrypoints.file" in result.stderr


def test_repository_context_graph() -> None:
    assert validate(Path(__file__).resolve().parents[1]) == []
