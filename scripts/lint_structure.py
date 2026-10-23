"""Validate the repository context graph without importing application code."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

TIERS = {"root", "domain", "service", "package", "tool"}
REQUIRED = {
    "version",
    "tier",
    "path",
    "parent",
    "summary",
    "responsibilities",
    "boundaries",
    "entrypoints",
    "dependencies",
    "submodules",
    "contracts",
    "tests",
}


@dataclass(frozen=True)
class Manifest:
    file: Path
    parent: Path | None
    children: tuple[Path, ...]


def mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label}: expected a mapping")
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{label}: keys must be strings")
    return cast(dict[str, object], raw)


def sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected a list")
    return cast(list[object], value)


def string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a nonempty string")
    return value


def reference(
    root: Path,
    base: Path,
    value: object,
    label: str,
    kind: Literal["file", "directory", "either"],
) -> Path:
    relative = Path(string(value, label))
    if relative.is_absolute():
        raise ValueError(f"{label}: must be relative")
    target = (base / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"{label}: escapes repository")
    if not target.exists():
        raise ValueError(f"{label}: missing path {relative}")
    if kind == "file" and not target.is_file():
        raise ValueError(f"{label}: expected a file")
    if kind == "directory" and not target.is_dir():
        raise ValueError(f"{label}: expected a directory")
    return target


def load_manifest(root: Path, file: Path) -> Manifest:
    data = mapping(
        cast(object, yaml.safe_load(file.read_text(encoding="utf-8"))), "manifest"
    )
    missing = REQUIRED - data.keys()
    if missing:
        raise ValueError(f"missing required keys: {', '.join(sorted(missing))}")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("version: expected integer 1")
    tier = string(data["tier"], "tier")
    if tier not in TIERS:
        raise ValueError(f"tier: unsupported value {tier}")
    directory = file.parent
    if reference(root, root, data["path"], "path", "directory") != directory:
        raise ValueError("path: does not match manifest directory")
    is_root = file == root / "structure.yml"
    parent: Path | None = None
    if is_root:
        if data["parent"] is not None or tier != "root":
            raise ValueError("root must have tier root and parent null")
    else:
        if tier == "root" or data["parent"] is None:
            raise ValueError("child must have a non-root tier and a parent")
        parent = reference(root, directory, data["parent"], "parent", "file")
        if parent.name != "structure.yml" or parent.parent not in directory.parents:
            raise ValueError("parent: must be an ancestor structure.yml")
    string(data["summary"], "summary")
    for key in ("responsibilities", "boundaries"):
        values = sequence(data[key], key)
        if not values:
            raise ValueError(f"{key}: must not be empty")
        for value in values:
            string(value, key)
    for value in sequence(data["entrypoints"], "entrypoints"):
        entry = mapping(value, "entrypoints")
        string(entry.get("name"), "entrypoints.name")
        string(entry.get("purpose"), "entrypoints.purpose")
        reference(root, directory, entry.get("file"), "entrypoints.file", "file")
    for value in sequence(data.get("files", []), "files"):
        # Optional custom inventory accepts strings or {path, ...metadata} records.
        path = value if isinstance(value, str) else mapping(value, "files").get("path")
        reference(root, directory, path, "files.path", "either")
    dependencies = mapping(data["dependencies"], "dependencies")
    for value in sequence(dependencies.get("internal"), "dependencies.internal"):
        reference(root, root, value, "dependencies.internal", "either")
    for value in sequence(dependencies.get("external"), "dependencies.external"):
        string(value, "dependencies.external")
    children: list[Path] = []
    names: set[str] = set()
    for value in sequence(data["submodules"], "submodules"):
        child = mapping(value, "submodules")
        name = string(child.get("name"), "submodules.name")
        string(child.get("summary"), "submodules.summary")
        child_dir = reference(
            root, directory, child.get("path"), "submodules.path", "directory"
        )
        target = reference(
            root, directory, child.get("manifest"), "submodules.manifest", "file"
        )
        if target != child_dir / "structure.yml" or directory not in child_dir.parents:
            raise ValueError(
                "submodules: manifest must belong to its descendant directory"
            )
        if target in children or name in names:
            raise ValueError("submodules: duplicate child or name")
        children.append(target)
        names.add(name)
    for value in sequence(data["contracts"], "contracts"):
        contract = mapping(value, "contracts")
        string(contract.get("name"), "contracts.name")
        reference(root, directory, contract.get("doc"), "contracts.doc", "file")
    tests = mapping(data["tests"], "tests")
    reference(root, directory, tests.get("directory"), "tests.directory", "directory")
    string(tests.get("command"), "tests.command")
    return Manifest(file, parent, tuple(children))


def discover(root: Path) -> list[Path]:
    """Git's index plus nonignored new files; never recursively scan media trees."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "structure.yml",
            "**/structure.yml",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted({root / name for name in result.stdout.split("\0") if name})


def validate(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    manifests: dict[Path, Manifest] = {}
    for file in discover(root):
        try:
            if file.is_symlink():
                raise ValueError("manifest must not be a symbolic link")
            manifests[file] = load_manifest(root, file)
        except (TypeError, ValueError, OSError, yaml.YAMLError) as exc:
            errors.append(f"{file.relative_to(root)}: {exc}")
    root_file = root / "structure.yml"
    if root_file not in manifests:
        errors.append("structure.yml: missing or invalid root manifest")
    callers: dict[Path, Path] = {}
    for manifest in manifests.values():
        for child in manifest.children:
            if child in callers:
                errors.append(f"{child.relative_to(root)}: multiple parents")
            callers[child] = manifest.file
            target = manifests.get(child)
            if target is None:
                errors.append(
                    f"{child.relative_to(root)}: child is invalid or ignored by Git"
                )
            elif target.parent != manifest.file:
                errors.append(
                    f"{child.relative_to(root)}: parent does not match caller"
                )
    for file, manifest in manifests.items():
        if file != root_file and callers.get(file) != manifest.parent:
            errors.append(
                f"{file.relative_to(root)}: parent does not link back to child"
            )
    reachable: set[Path] = set()
    pending = [root_file]
    while pending:
        file = pending.pop()
        if file in reachable or file not in manifests:
            continue
        reachable.add(file)
        pending.extend(manifests[file].children)
    for file in manifests.keys() - reachable:
        errors.append(f"{file.relative_to(root)}: unreachable from root")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    try:
        errors = validate(cast(Path, args.root))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"structure validation failed: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Structure manifests: schema, references, and root reachability valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
