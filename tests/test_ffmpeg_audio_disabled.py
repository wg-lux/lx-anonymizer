import ast
from pathlib import Path


def test_every_literal_ffmpeg_command_disables_audio() -> None:
    package_root = Path(__file__).parents[1] / "lx_anonymizer"
    missing_audio_disable: list[str] = []

    for source_path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=source_path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.List) or not node.elts:
                continue
            first = node.elts[0]
            if not (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value == "ffmpeg"
            ):
                continue
            string_arguments = {
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
            if "-an" not in string_arguments:
                relative_path = source_path.relative_to(package_root.parent)
                missing_audio_disable.append(f"{relative_path}:{node.lineno}")

    assert missing_audio_disable == [], "FFmpeg commands missing -an: " + ", ".join(
        missing_audio_disable
    )
