from lx_anonymizer._native import native
from lx_anonymizer.ocr import ocr_frame_tesserocr as ocr_mod


def test_native_or_python_normalize_matches_reference():
    samples = [
        "",
        "Patient:::   Müller",
        "E 15702/2024  09:53:32",
        "abc....,,,:::",
        "äöüß -- ok",
    ]
    for sample in samples:
        assert ocr_mod._normalize_ocr_text_impl(
            sample
        ) == ocr_mod._py_normalize_ocr_text(sample)


def test_native_or_python_gibberish_matches_reference():
    samples = [
        "",
        "##@@",
        "Patient Müller",
        "09:53:32",
        "E 15702/2024",
        "zzzx qwrty",
    ]
    for sample in samples:
        assert ocr_mod._is_gibberish_impl(sample) == ocr_mod._py_is_gibberish(sample)
        assert ocr_mod._looks_structured_overlay_text_impl(
            sample
        ) == ocr_mod._py_looks_structured_overlay_text(sample)


def test_native_or_python_candidate_rank_matches_reference():
    samples = [
        ("Patient Müller", 87.5),
        ("09:53:32", 92.0),
        ("", 0.0),
        ("##@@", 30.0),
    ]
    for text, conf in samples:
        assert ocr_mod._candidate_rank_impl(text, conf) == ocr_mod._py_candidate_rank(
            text, conf
        )


def test_native_fuzzy_match_best_prefers_expected_candidate_when_available():
    if native is None:
        return

    best_match, ratio = native.fuzzy_match_best(
        "Patient John Doe",
        ["Patlent John Doe", "Procedure Notes", "Random Value"],
        0.5,
    )

    assert best_match == "Patlent John Doe"
    assert ratio >= 0.5
