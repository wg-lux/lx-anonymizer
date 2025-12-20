#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ---------- Helpers ----------

def human(n: int) -> str:
    for unit in ["B","K","M","G","T"]:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}P"

def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except Exception:
            pass
    return total

def list_files(path: Path) -> List[Path]:
    files: List[Path] = []
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                files.append(p)
        except Exception:
            pass
    return files

def delete_until_target(paths: List[Path], target_bytes: int, dry_run: bool) -> Tuple[int, List[Path]]:
    """
    Delete oldest files first (by mtime) until total size <= target_bytes.
    Returns (freed_bytes, deleted_files).
    """
    files = [p for p in paths if p.exists()]
    files.sort(key=lambda p: p.stat().st_mtime)  # oldest first
    total = sum(p.stat().st_size for p in files)
    if total <= target_bytes:
        return 0, []

    freed = 0
    deleted: List[Path] = []
    for p in files:
        if total - freed <= target_bytes:
            break
        try:
            sz = p.stat().st_size
            if not dry_run:
                p.unlink(missing_ok=True)
            freed += sz
            deleted.append(p)
        except Exception:
            pass
    return freed, deleted

def load_dotenv(env_path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def pick_first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p and p.exists():
            return p
    return None

# ---------- Config ----------

@dataclass
class Paths:
    project_root: Path
    env_path: Path
    hf_home: Path
    hf_hub_cache: Path
    HF_HOME: Path
    ollama_models: Path
    pip_cache: Path
    torch_cache: Path
    spacy_cache: Path
    trash_local: Path

def resolve_paths() -> Paths:
    # Projektroot wie in deinem env_setup.py: two parents up from scripts dir
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"

    env_file = load_dotenv(env_path)
    env_get = lambda key, default: os.environ.get(key, env_file.get(key, default))

    home = Path.home()
    hf_home = Path(env_get("HF_HOME", home / ".cache" / "huggingface"))
    hf_hub_cache = Path(env_get("HF_HUB_CACHE", hf_home / "hub"))
    HF_HOME = Path(env_get("HF_HOME", hf_hub_cache))
    ollama_models = Path(env_get("OLLAMA_MODELS", home / ".ollama" / "models"))

    pip_cache = Path(os.environ.get("PIP_CACHE_DIR", home / ".cache" / "pip"))
    torch_cache = Path(home / ".cache" / "torch")
    spacy_cache = Path(home / ".cache" / "spacy")
    trash_local = Path(home / ".local" / "share" / "Trash" / "files")

    return Paths(
        project_root=project_root,
        env_path=env_path,
        hf_home=hf_home,
        hf_hub_cache=hf_hub_cache,
        HF_HOME=HF_HOME,
        ollama_models=ollama_models,
        pip_cache=pip_cache,
        torch_cache=torch_cache,
        spacy_cache=spacy_cache,
        trash_local=trash_local,
    )

# ---------- Cleanup Routines ----------

def lru_trim_dir(dir_path: Path, limit_gb: float, dry_run: bool, label: str) -> int:
    if not dir_path.exists():
        print(f"[{label}] {dir_path} existiert nicht — übersprungen.")
        return 0
    total = dir_size(dir_path)
    limit = int(limit_gb * (1024**3))
    print(f"[{label}] Aktuell: {human(total)}  Limit: {limit_gb:.1f}G  Pfad: {dir_path}")
    if total <= limit:
        print(f"[{label}] OK — unter Limit.")
        return 0
    files = list_files(dir_path)
    freed, deleted = delete_until_target(files, limit, dry_run)
    print(f"[{label}] {'Würde ' if dry_run else ''}löschen: {len(deleted)} Dateien  |  Freigegeben: {human(freed)}")
    return freed

def clean_temp_dirs(paths: List[Path], max_age_days: int, dry_run: bool, label: str) -> int:
    now = time.time()
    cutoff = now - max_age_days * 86400
    freed = 0
    count = 0
    for p in paths:
        if not p.exists():
            continue
        for f in p.rglob("*"):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    sz = f.stat().st_size
                    if not dry_run:
                        f.unlink(missing_ok=True)
                    freed += sz
                    count += 1
            except Exception:
                pass
    print(f"[{label}] {'Würde ' if dry_run else ''}löschen: {count} alte Dateien  |  Freigegeben: {human(freed)}")
    return freed

def journalctl_vacuum(max_size: str, dry_run: bool):
    # Optional, nur wenn systemd-journal vorhanden ist
    cmd = ["journalctl", f"--vacuum-size={max_size}"]
    print(f"[LOG] {'Würde ausführen' if dry_run else 'Ausführen'}: {' '.join(cmd)}")
    if not dry_run:
        try:
            subprocess.run(cmd, check=False)
        except FileNotFoundError:
            print("[LOG] journalctl nicht gefunden — übersprungen.")

def apt_dnf_pacman_clean(dry_run: bool):
    # Versuche distrounabhängig zu sein: nur ausführen, wenn vorhanden
    candidates = [
        (["apt-get", "clean"], "APT clean"),
        (["dnf", "clean", "all"], "DNF clean"),
        (["pacman", "-Sc", "--noconfirm"], "Pacman clean"),
    ]
    for cmd, label in candidates:
        exe = shutil.which(cmd[0])
        if exe:
            print(f"[PKG] {'Würde ' if dry_run else ''}{label}")
            if not dry_run:
                subprocess.run(cmd, check=False)

def ollama_prune(dry_run: bool):
    exe = shutil.which("ollama")
    if not exe:
        print("[Ollama] ollama nicht gefunden — übersprungen.")
        return
    cmd = ["ollama", "prune", "-f"]
    print(f"[Ollama] {'Würde ausführen' if dry_run else 'Ausführen'}: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=False)

def hf_cli_scan_cache(hf_dir: Path, dry_run: bool):
    """
    Optional: Nutze offizielle CLI, wenn vorhanden (löscht unreferenzierte Blobs).
    Wir führen nur 'scan-cache' aus und zeigen Summary an. Löschen weiterhin LRU-basiert oben,
    außer du möchtest das hier aktivieren.
    """
    exe = shutil.which("huggingface-cli")
    if not exe or not hf_dir.exists():
        return
    cmd = ["huggingface-cli", "scan-cache", "--dir", str(hf_dir)]
    print(f"[HF CLI] {'Würde ausführen' if dry_run else 'Ausführen'}: {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=False)

# ---------- spaCy Optimierungen ----------

SPACY_MODEL_CANDIDATES = [
    "de_core_news_lg",
    "de_core_news_md",
    "de_core_news_sm",
    "de_dep_news_trf",  # Transformer-Variante
    "core_news_de",     # dein Alias (Fallback)
]

SPACY_TUNE_TEMPLATE = r"""
# Beispiel: Nutzung mit Optimierungen
import spacy

def load_de(max_len:int=2_000_000, disable:list=None, use_gpu:bool=True):
    disable = disable or []  # z.B. ["ner","lemmatizer"] wenn nicht benötigt
    if use_gpu:
        try:
            spacy.prefer_gpu()  # nutzt CuPy/torch wenn vorhanden
        except Exception:
            pass
    model = None
    for name in {cands}:
        try:
            model = spacy.load(name, disable=disable)
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError("Kein deutsches spaCy-Modell gefunden. Bitte `pip install de_core_news_sm` o.ä.")
    # Größere Texte erlauben
    model.max_length = max_len
    return model

# Beispiel-Inferenz mit Batch- und Prozess-Tuning
def pipe_texts(nlp, texts, batch_size:int=200, n_process:int=2):
    # select_pipes reduziert Overhead, wenn du nur Tokenisierung/Tagging brauchst
    with nlp.select_pipes(disable=["ner"]):  # passe an deinen Use-Case an
        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            yield doc
"""


def main():
    ap = argparse.ArgumentParser(description="LRU-Cleanup für HF/Transformers/Ollama/PIP/Torch/Trash + spaCy-Hints")
    ap.add_argument("--apply", action="store_true", help="Tatsächlich löschen (ohne: Dry-Run/Report)")
    ap.add_argument("--hf-limit-gb", type=float, default=40.0, help="Limit für HF_HUB_CACHE/HF_HOME (GB)")
    ap.add_argument("--pip-limit-gb", type=float, default=5.0, help="Limit für PIP-Cache (GB)")
    ap.add_argument("--torch-limit-gb", type=float, default=10.0, help="Limit für Torch-Cache (GB)")
    ap.add_argument("--spacy-limit-gb", type=float, default=2.0, help="Limit für spaCy-Cache (GB)")
    ap.add_argument("--tmp-max-age-days", type=int, default=14, help="Max Alter für /tmp & Trash in Tagen")
    ap.add_argument("--vacuum-journal", default="300M", help="journalctl --vacuum-size=... (z.B. 300M, 1G)")
    args = ap.parse_args()

    dry = not args.apply
    paths = resolve_paths()

    print(f"Projektroot: {paths.project_root}")
    print(f".env:        {paths.env_path} (wird gelesen, falls vorhanden)")
    print(f"Mode:        {'Dry-Run' if dry else 'APPLY'}")
    print("—"*72)

    total_freed = 0

    # Hugging Face / Transformers – je nach Setup identisch
    # Hinweis: Wenn beide auf dasselbe Verzeichnis zeigen, trimme nur einmal.
    hf_dirs = []
    if paths.hf_hub_cache.exists():
        hf_dirs.append(("HF_HUB_CACHE", paths.hf_hub_cache))
    if paths.HF_HOME.exists() and paths.HF_HOME.resolve() != paths.hf_hub_cache.resolve():
        hf_dirs.append(("HF_HOME", paths.HF_HOME))

    seen: set[Path] = set()
    for label, d in hf_dirs:
        real = d.resolve()
        if real in seen:
            continue
        seen.add(real)
        total_freed += lru_trim_dir(real, args.hf_limit_gb, dry, label)
        hf_cli_scan_cache(paths.hf_hub_cache, dry)

    # PIP / Torch / spaCy
    total_freed += lru_trim_dir(paths.pip_cache, args.pip_limit_gb, dry, "PIP")
    total_freed += lru_trim_dir(paths.torch_cache, args.torch_limit_gb, dry, "TORCH")
    total_freed += lru_trim_dir(paths.spacy_cache, args.spacy_limit_gb, dry, "SPACY")

    # Trash & /tmp (alte Dateien)
    tmp_candidates = [Path("/tmp"), paths.trash_local]
    total_freed += clean_temp_dirs(tmp_candidates, args.tmp_max_age_days, dry, "TMP/TRASH")

    # Paketmanager-Caches (wenn Tool vorhanden)
    apt_dnf_pacman_clean(dry)

    # systemd-Journal verkleinern
    journalctl_vacuum(args.vacuum_journal, dry)

    # Ollama ungenutzte Layer prune
    ollama_prune(dry)

    print("—"*72)
    print(f"Gesamt {'potenziell ' if dry else ''}freigegeben: {human(total_freed)}")
    if dry:
        print("Dry-Run beendet. Mit '--apply' wirklich löschen.")

if __name__ == "__main__":
    main()
