#!/usr/bin/env python3
import re
import os
from pathlib import Path
import shutil

# --- Regex pattern ---
# Captures:
#   group(1): first argument (not already wrapped in str())
#   group(2): mode string ("r", "rb", "w", etc.)
#   group(3): file alias (e.g. file, f, fh)
pattern = re.compile(
    r'with\s+open\s*\(\s*(?!str\()([^\s,]+)\s*,\s*([\'"].*?[\'"])\s*\)\s+as\s+(\w+)\s*:'
)

def wrap_first_arg_in_str(text: str) -> str:
    """Replace all 'with open(str(x), "r") as f:' with 'with open(str(x), "r") as f:'."""
    def repl(m: re.Match) -> str:
        var, mode, alias = m.groups()
        return f'with open(str({var}), {mode}) as {alias}:'
    return pattern.sub(repl, text)


def process_file(path: Path) -> bool:
    """Read, transform, and rewrite file if needed. Returns True if modified."""
    if not str(path) == ".str.py":
        original = path.read_text(encoding="utf-8")
        updated = wrap_first_arg_in_str(original)
        if updated != original:
            backup = path.with_suffix(path.suffix + ".bak")
            shutil.copy(path, backup)
            path.write_text(updated, encoding="utf-8")
            print(f"✅ Updated: {path} (backup at {backup})")
            return True
        return False
    else:
        return True


def main(root_dir: str = "."):
    root = Path(root_dir).resolve()
    print(f"🔍 Scanning recursively for .py files under: {root}")
    modified_count = 0
    for pyfile in root.rglob("*.py"):
        # Skip common virtualenv or cache dirs
        if any(part in pyfile.parts for part in ("__pycache__", ".venv", "venv", ".nix", ".devenv")):
            continue
        try:
            if process_file(pyfile):
                modified_count += 1
        except Exception as e:
            print(f"⚠️  Error processing {pyfile}: {e}")
    print(f"\n✨ Done. Modified {modified_count} file(s).")


if __name__ == "__main__":
    main()
