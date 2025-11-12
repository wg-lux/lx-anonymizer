from faker import Faker
from datetime import datetime, timedelta
import random
import re
from hashlib import sha256
from typing import Dict, Tuple, Optional, Iterable
from .custom_logger import get_logger


def replace_umlauts(text):
    """Replace German umlauts with their corresponding letter (ä -> ae)."""
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('Ä', 'AE')
    text = text.replace('Ö', 'OE')
    text = text.replace('Ü', 'UE')
    return text

def replace_large_numbers(text):
    """Replaces all numbers with at least 5 digits with random numbers of the same length."""
    def random_number(n):
        return ''.join([str(random.randint(0, 9)) for _ in range(n)])
    
    numbers_to_replace = re.findall(r'\b\d{5,}\b', text)
    
    for number in numbers_to_replace:
        length = len(number)
        random_num = random_number(length)
        text = text.replace(number, random_num)

    return text

def remove_titles(name):
    """Remove titles like 'Dr.' and 'Dr. med.' from names."""
    return re.sub(r'(Dr\. med\. |Dr\. |Prof\.)', '', name)

def cutoff_leading_text(text, cutoff_flags):
    """
    Remove all text above the first occurrence of any cutoff flag.
    Used primarily for removing letterheads in medical reports.
    """
    if not cutoff_flags:  # If no flags provided, return text as is
        return text
        
    lines = text.split('\n')
    for i, line in enumerate(lines):
        for flag in cutoff_flags:
            if flag in line:
                return '\n'.join(lines[i:])
    return text

def cutoff_trailing_text(text, cutoff_flags):
    """
    Remove all text below the first occurrence of any cutoff flag.
    Used primarily for removing footers in medical reports.
    """
    if not cutoff_flags:  # If no flags provided, return text as is
        return text
        
    lines = text.split('\n')
    for i, line in enumerate(lines):
        for flag in cutoff_flags:
            if flag in line:
                return '\n'.join(lines[:i+1])
    return text

def replace_employee_names(text, first_names, last_names, locale=None):
    """Replace known employee names with fake names."""
    fake = Faker(locale=locale)
    if first_names:
        for first_name in first_names:
            if first_name and first_name in text:
                text = text.replace(first_name, fake.first_name())
    
    if last_names:
        for last_name in last_names:
            if last_name and last_name in text:
                text = text.replace(last_name, fake.last_name())
                
    return text



def _seeded_faker(seed: str, locale: str) -> Faker:
    rnd = int(sha256(seed.encode("utf-8")).hexdigest(), 16) % (2**31)
    fake = Faker(locale=locale)
    Faker.seed(rnd)
    random.seed(rnd)
    return fake

def _preserve_case_like(sample: str, replacement: str) -> str:
    if sample.isupper():
        return replacement.upper()
    if sample.istitle():
        return replacement.title()
    if sample.islower():
        return replacement.lower()
    # mixed or unknown -> return as given
    return replacement

def _compile_name_patterns(name: str) -> Iterable[re.Pattern]:
    """
    Build robust regex patterns for a name:
    - exact word-boundary match
    - tolerate repeated spaces or hyphens in OCR (e.g., 'Van  der-Meer')
    """
    if not name:
        return []
    # Escape and allow variable spaces/hyphens between tokens
    tokens = [re.escape(t) for t in re.split(r"[\s\-]+", name.strip()) if t]
    if not tokens:
        return []
    flexible = r"[ \-]+".join(tokens)
    # \b with Unicode words works for German letters in Python's regex engine
    return [
        re.compile(rf"\b({flexible})\b", flags=re.IGNORECASE),
    ]

def _safe_sub_all(text: str, patterns: Iterable[re.Pattern], repl_factory) -> str:
    """Apply multiple compiled patterns with a case-preserving callable replacement."""
    for pat in patterns:
        def _repl(m: re.Match) -> str:
            return repl_factory(m.group(0))
        text = pat.sub(_repl, text)
    return text

def _shift_date_iso(iso: str, days_window: Tuple[int, int], seed: str) -> Optional[str]:
    """Deterministic shift of an ISO date (YYYY-MM-DD) within a window."""
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d").date()
    except Exception:
        return None
    a, b = days_window
    # deterministic offset from seed + iso
    rnd = int(sha256((seed + iso).encode("utf-8")).hexdigest(), 16)
    delta = a + (rnd % (b - a + 1))
    return (dt + timedelta(days=delta)).strftime("%Y-%m-%d")

def _date_string_variants(iso: str, fmt: str) -> Tuple[str, str]:
    """
    Return two common written forms: localized format and ISO.
    (We replace both to be safe.)
    """
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d")
    except Exception:
        return iso, iso
    return dt.strftime(fmt), dt.strftime("%Y-%m-%d")

def anonymize_text(
    text: str,
    report_meta: Dict,
    text_date_format: str = "%d.%m.%Y",
    lower_cut_off_flags: Optional[Iterable[str]] = None,
    upper_cut_off_flags: Optional[Iterable[str]] = None,
    locale: str = "de_DE",
    first_names: Optional[Iterable[str]] = None,
    last_names: Optional[Iterable[str]] = None,
    apply_cutoffs: bool = False,
    verbose: bool = False,
    *,
    anonymize_dates: bool = True,
) -> str:
    """
    Safer, deterministic anonymization:
    - Word-boundary, case-preserving name replacements
    - Deterministic pseudonyms per document
    - Optional deterministic date shifting (DOB, exam date)
    - Employee name replacements
    - Cutoff (upper/lower) applied early
    """
    logger = get_logger(__name__, verbose=verbose)
    if not isinstance(text, str) or not text:
        return text

    # 1) Apply cutoffs first (less work, fewer false positives later)
    if apply_cutoffs:
        if upper_cut_off_flags:
            text = cutoff_leading_text(text, upper_cut_off_flags)
        if lower_cut_off_flags:
            text = cutoff_trailing_text(text, lower_cut_off_flags)

    # Build a deterministic seed from available meta fields
    seed_material = "|".join(
        str(report_meta.get(k, "")) for k in (
            "pdf_hash", "file_path", "patient_first_name", "patient_last_name",
            "casenumber", "examination_date", "patient_dob"
        )
    ) or text[:256]  # fallback to text prefix if meta is sparse
    fake = _seeded_faker(seed_material, locale)

    # 2) Patient names (robust, handles presence of only one of them)
    pf = (report_meta.get("patient_first_name") or "").strip()
    pl = (report_meta.get("patient_last_name") or "").strip()

    # Generate deterministic pseudonyms
    pseudo_pf = fake.first_name() if pf else None
    pseudo_pl = fake.last_name() if pl else None

    # Replace "First Last" (combined) before single parts to avoid partial breaks
    if pf and pl:
        full = f"{pf} {pl}"
        patns = _compile_name_patterns(full)
        text = _safe_sub_all(
            text, patns,
            lambda s: _preserve_case_like(s, f"{pseudo_pf} {pseudo_pl}")
        )

    # Replace last name alone
    if pl:
        patns = _compile_name_patterns(pl)
        text = _safe_sub_all(
            text, patns,
            lambda s: _preserve_case_like(s, pseudo_pl or fake.last_name())
        )

    # Replace first name alone
    if pf:
        patns = _compile_name_patterns(pf)
        text = _safe_sub_all(
            text, patns,
            lambda s: _preserve_case_like(s, pseudo_pf or fake.first_name())
        )

    # 3) Examiner names (if present)
    ef = (report_meta.get("examiner_first_name") or "").strip()
    el = (report_meta.get("examiner_last_name") or "").strip()
    pseudo_ef = fake.first_name() if ef else None
    pseudo_el = fake.last_name() if el else None

    if ef and el:
        full = f"{ef} {el}"
        patns = _compile_name_patterns(full)
        text = _safe_sub_all(
            text, patns,
            lambda s: _preserve_case_like(s, f"{pseudo_ef} {pseudo_el}")
        )

    if el:
        patns = _compile_name_patterns(el)
        text = _safe_sub_all(
            text, patns,
            lambda s: _preserve_case_like(s, pseudo_el or fake.last_name())
        )

    if ef:
        patns = _compile_name_patterns(ef)
        text = _safe_sub_all(
            text, patns,
            lambda s: _preserve_case_like(s, pseudo_ef or fake.first_name())
        )

    # 4) Replace configured employee names (uses simple .replace – acceptable for non-critical staff names)
    if first_names or last_names:
        text = replace_employee_names(text, first_names, last_names, locale=locale)

    # 5) Large numbers (IDs, device serials, etc.)
    text = replace_large_numbers(text)

    # 6) Dates (DOB / examination date) — deterministic shift, preserve visual format
    if anonymize_dates:
        # Patient DOB
        dob_iso = None
        dob_val = report_meta.get("patient_dob")
        if isinstance(dob_val, str):
            # Assume ISO or already formatted; try normalize to ISO
            m = re.match(r"^\d{4}-\d{2}-\d{2}$", dob_val)
            dob_iso = dob_val if m else None
        elif isinstance(dob_val, (datetime, )):
            dob_iso = dob_val.strftime("%Y-%m-%d")

        # Accept already-cleaned string dates in  YYYY-MM-DD from upstream
        if dob_iso:
            fake_dob_iso = _shift_date_iso(dob_iso, (-1200, -365), seed_material)  # keep age-ish but not exact
            if fake_dob_iso:
                dob_local, dob_iso_str = _date_string_variants(dob_iso, text_date_format)
                fake_local, fake_iso_str = _date_string_variants(fake_dob_iso, text_date_format)
                # Replace both local and ISO spelling if present
                for original, fake_str in [(dob_local, fake_local), (dob_iso_str, fake_iso_str)]:
                    if original and fake_str and original in text:
                        text = re.sub(re.escape(original), fake_str, text)

        # Examination date
        ex_iso = None
        ex_val = report_meta.get("examination_date")
        if isinstance(ex_val, str):
            m = re.match(r"^\d{4}-\d{2}-\d{2}$", ex_val)
            ex_iso = ex_val if m else None
        elif isinstance(ex_val, (datetime, )):
            ex_iso = ex_val.strftime("%Y-%m-%d")

        if ex_iso:
            fake_ex_iso = _shift_date_iso(ex_iso, (-30, 30), seed_material)  # small temporal jitter
            if fake_ex_iso:
                ex_local, ex_iso_str = _date_string_variants(ex_iso, text_date_format)
                fake_local, fake_iso_str = _date_string_variants(fake_ex_iso, text_date_format)
                for original, fake_str in [(ex_local, fake_local), (ex_iso_str, fake_iso_str)]:
                    if original and fake_str and original in text:
                        text = re.sub(re.escape(original), fake_str, text)

    return text
