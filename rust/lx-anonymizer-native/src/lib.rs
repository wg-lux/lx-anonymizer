#![allow(unsafe_op_in_unsafe_fn)]

use numpy::PyReadonlyArray3;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use regex::Regex;
use std::cmp::Ordering;
use std::sync::OnceLock;
use strsim::jaro_winkler;

const EXPECTED_OCR_CHARS: &str =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß0123456789 .,:;/-()[]";
const OCR_VOWELS: &str = "aeiouäöüAEIOUÄÖÜ";

fn time_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\d{1,2}:\d{2}(?::\d{2})?").expect("valid time regex"))
}

fn date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]\d{4}")
            .expect("valid date regex")
    })
}

fn case_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Z]\s*\d{4,}/\d{4}").expect("valid case regex"))
}

fn compact_code_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b[A-Z]\s*\d{5,}\b|\b[A-Z]\d{5,}\b").expect("valid compact code regex")
    })
}

fn device_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\d{8,}").expect("valid device regex"))
}

fn ratio_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b\d+(?:[.,]\d+)?/\d+(?:[.,]\d+)?\b").expect("valid ratio regex")
    })
}

fn normalize_filter_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\w\s.,:;/-ÄÖÜäöüß]").expect("valid normalize filter regex"))
}

fn repeated_punct_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[.,:;]{2,}").expect("valid punctuation cleanup regex"))
}

fn multispace_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s{2,}").expect("valid multispace regex"))
}

fn looks_structured_inner(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    time_re().is_match(text)
        || date_re().is_match(text)
        || case_re().is_match(text)
        || compact_code_re().is_match(text)
        || device_re().is_match(text)
        || ratio_re().is_match(text)
}

fn contains_vowel(word: &str) -> bool {
    word.chars().any(|c| OCR_VOWELS.contains(c))
}

fn normalize_text_inner(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let filtered = normalize_filter_re().replace_all(text, "");
    let mut collapsed = String::with_capacity(filtered.len());
    let mut prev: Option<char> = None;
    for ch in filtered.chars() {
        let collapseable = matches!(ch, '.' | ',' | ':' | ';');
        if !(collapseable && prev == Some(ch)) {
            collapsed.push(ch);
        }
        prev = Some(ch);
    }
    let collapsed = repeated_punct_re().replace_all(&collapsed, |caps: &regex::Captures| {
        caps.get(0)
            .and_then(|m| m.as_str().chars().next())
            .map(|c| c.to_string())
            .unwrap_or_default()
    });
    let normalized = multispace_re().replace_all(&collapsed, " ");
    normalized.trim().to_string()
}

fn gibberish_score_inner(text: &str) -> f64 {
    if text.is_empty() {
        return 1.0;
    }

    let length = text.chars().count().max(1) as f64;
    let mut score = 0.0;

    let alpha_ratio = text.chars().filter(|c| c.is_alphabetic()).count() as f64 / length;
    if alpha_ratio < 0.2 {
        score += 0.35;
    } else if alpha_ratio < 0.35 {
        score += 0.15;
    }

    let nonstandard_ratio = text
        .chars()
        .filter(|c| !EXPECTED_OCR_CHARS.contains(*c))
        .count() as f64
        / length;
    score += (nonstandard_ratio * 0.8).min(0.4);

    let punct_like = text
        .chars()
        .filter(|c| !c.is_alphanumeric() && !c.is_whitespace() && !".,:;/-".contains(*c))
        .count() as f64;
    score += ((punct_like / length) * 0.6).min(0.2);

    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| w.chars().count() > 1)
        .collect();
    if !words.is_empty() {
        let vowel_words = words.iter().filter(|w| contains_vowel(w)).count() as f64;
        let ratio = vowel_words / words.len() as f64;
        if ratio < 0.15 {
            score += 0.25;
        } else if ratio < 0.3 {
            score += 0.1;
        }
    }

    score.clamp(0.0, 1.0)
}

fn is_gibberish_inner(text: &str) -> bool {
    if text.chars().count() < 3 {
        return true;
    }

    if looks_structured_inner(text) {
        return false;
    }

    let length = text.chars().count().max(1) as f64;
    let alpha_ratio = text.chars().filter(|c| c.is_alphabetic()).count() as f64 / length;
    if alpha_ratio < 0.20 {
        return true;
    }

    let nonstandard = text
        .chars()
        .filter(|c| !EXPECTED_OCR_CHARS.contains(*c))
        .count() as f64;
    if nonstandard > 0.4 * length {
        return true;
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return true;
    }

    let multi_char_words: Vec<&str> = words
        .iter()
        .copied()
        .filter(|w| w.chars().count() > 1)
        .collect();
    if !multi_char_words.is_empty() {
        let words_with_vowels = multi_char_words
            .iter()
            .filter(|w| contains_vowel(w))
            .count() as f64;
        if words_with_vowels < 0.15 * multi_char_words.len() as f64 {
            return true;
        }
    }

    let unique_chars = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<std::collections::BTreeSet<_>>()
        .len() as f64;
    unique_chars < length * 0.1
}

type BoxTuple = (i32, i32, i32, i32);
type OcrEntry = (String, BoxTuple);

fn mean_color_for_box(
    image: &PyReadonlyArray3<'_, u8>,
    box_region: Option<BoxTuple>,
) -> (u8, u8, u8) {
    let array = image.as_array();
    let shape = array.shape();
    if shape.len() != 3 || shape[2] < 3 {
        return (255, 255, 255);
    }

    let (height, width) = (shape[0] as i32, shape[1] as i32);
    let (start_x, start_y, end_x, end_y) = match box_region {
        Some((sx, sy, ex, ey)) => (
            sx.clamp(0, width),
            sy.clamp(0, height),
            ex.clamp(0, width),
            ey.clamp(0, height),
        ),
        None => (0, 0, width, height),
    };

    if start_x >= end_x || start_y >= end_y {
        return (255, 255, 255);
    }

    let mut count: u64 = 0;
    let mut sum0: u64 = 0;
    let mut sum1: u64 = 0;
    let mut sum2: u64 = 0;

    for y in start_y..end_y {
        for x in start_x..end_x {
            sum0 += u64::from(array[[y as usize, x as usize, 0]]);
            sum1 += u64::from(array[[y as usize, x as usize, 1]]);
            sum2 += u64::from(array[[y as usize, x as usize, 2]]);
            count += 1;
        }
    }

    if count == 0 {
        return (255, 255, 255);
    }

    (
        (sum0 / count) as u8,
        (sum1 / count) as u8,
        (sum2 / count) as u8,
    )
}

fn color_distance(left: (u8, u8, u8), right: (u8, u8, u8)) -> f64 {
    let dr = f64::from(left.0) - f64::from(right.0);
    let dg = f64::from(left.1) - f64::from(right.1);
    let db = f64::from(left.2) - f64::from(right.2);
    (dr * dr + dg * dg + db * db).sqrt()
}

#[pyfunction]
fn filter_empty_boxes_native(
    ocr_results: Vec<OcrEntry>,
    min_text_len: Option<usize>,
) -> Vec<OcrEntry> {
    let min_text_len = min_text_len.unwrap_or(2);
    ocr_results
        .into_iter()
        .filter(|(text, _)| text.trim().chars().count() >= min_text_len)
        .collect()
}

#[pyfunction]
fn combine_boxes_native(text_with_boxes: Vec<OcrEntry>, y_tolerance: Option<i32>) -> Vec<OcrEntry> {
    let y_tolerance = y_tolerance.unwrap_or(10);
    if text_with_boxes.is_empty() {
        return text_with_boxes;
    }

    let mut sorted_items = text_with_boxes;
    sorted_items.sort_by_key(|(_, bbox)| (bbox.1, bbox.0));

    let mut merged: Vec<OcrEntry> = vec![sorted_items[0].clone()];

    for (current_text, current_box) in sorted_items.into_iter().skip(1) {
        let (last_text, last_box) = merged.last_mut().expect("merged has first item");
        let (l_sx, l_sy, l_ex, l_ey) = *last_box;
        let (c_sx, c_sy, c_ex, c_ey) = current_box;

        if (l_sy - c_sy).abs() <= y_tolerance && (c_sx - l_ex) <= 10 {
            *last_box = (
                l_sx.min(c_sx),
                l_sy.min(c_sy),
                l_ex.max(c_ex),
                l_ey.max(c_ey),
            );
            *last_text = format!("{last_text} {current_text}");
        } else {
            merged.push((current_text, current_box));
        }
    }

    merged
}

#[pyfunction]
fn close_to_box_native(name_box: BoxTuple, phrase_box: BoxTuple) -> bool {
    (name_box.0 - phrase_box.0).abs() <= 10 && (name_box.1 - phrase_box.1).abs() <= 10
}

#[pyfunction]
fn make_box_from_device_list_native(x: i32, y: i32, w: i32, h: i32) -> BoxTuple {
    (x, y, x + w, y + h)
}

#[pyfunction]
fn find_or_create_close_box_native(
    phrase_box: BoxTuple,
    boxes: Vec<BoxTuple>,
    image_width: i32,
    min_offset: Option<i32>,
) -> BoxTuple {
    let min_offset = min_offset.unwrap_or(20);
    let (start_x, start_y, end_x, end_y) = phrase_box;
    let mut same_line_boxes = boxes
        .into_iter()
        .filter(|bbox| (bbox.1 - start_y).abs() <= 10)
        .collect::<Vec<_>>();

    let box_width = end_x - start_x;
    let required_offset = (box_width + min_offset).max(min_offset);

    if !same_line_boxes.is_empty() {
        same_line_boxes.sort_by_key(|bbox| bbox.0);
        for bbox in same_line_boxes {
            if bbox.0 >= end_x + required_offset {
                return bbox;
            }
        }
    }

    let new_start_x = (end_x + required_offset).min(image_width - box_width);
    let new_end_x = (new_start_x + box_width).min(image_width);
    (new_start_x, start_y, new_end_x, end_y)
}

#[pyfunction]
fn get_dominant_color_native(
    image: PyReadonlyArray3<'_, u8>,
    box_region: Option<BoxTuple>,
) -> (u8, u8, u8) {
    mean_color_for_box(&image, box_region)
}

#[pyfunction]
fn extend_boxes_if_needed_native(
    image: PyReadonlyArray3<'_, u8>,
    boxes: Vec<BoxTuple>,
    extension_margin: Option<i32>,
    color_threshold: Option<f64>,
) -> Vec<BoxTuple> {
    let extension_margin = extension_margin.unwrap_or(10);
    let color_threshold = color_threshold.unwrap_or(30.0);
    let shape = image.as_array().shape().to_vec();
    let image_height = shape.first().copied().unwrap_or(0) as i32;
    let image_width = shape.get(1).copied().unwrap_or(0) as i32;
    let mut extended_boxes = Vec::with_capacity(boxes.len());

    for bbox in boxes {
        let (mut start_x, mut start_y, mut end_x, mut end_y) = bbox;
        let dominant_color = mean_color_for_box(&image, Some(bbox));

        if start_y - extension_margin > 0 {
            let upper_color = mean_color_for_box(
                &image,
                Some((start_x, start_y - extension_margin, end_x, start_y)),
            );
            if color_distance(upper_color, dominant_color) > color_threshold {
                start_y = (start_y - extension_margin).max(0);
            }
        }

        if end_y + extension_margin < image_height {
            let lower_color = mean_color_for_box(
                &image,
                Some((start_x, end_y, end_x, end_y + extension_margin)),
            );
            if color_distance(lower_color, dominant_color) > color_threshold {
                end_y = (end_y + extension_margin).min(image_height);
            }
        }

        if start_x - extension_margin > 0 {
            let left_color = mean_color_for_box(
                &image,
                Some((start_x - extension_margin, start_y, start_x, end_y)),
            );
            if color_distance(left_color, dominant_color) > color_threshold {
                start_x = (start_x - extension_margin).max(0);
            }
        }

        if end_x + extension_margin < image_width {
            let right_color = mean_color_for_box(
                &image,
                Some((end_x, start_y, end_x + extension_margin, end_y)),
            );
            if color_distance(right_color, dominant_color) > color_threshold {
                end_x = (end_x + extension_margin).min(image_width);
            }
        }

        extended_boxes.push((start_x, start_y, end_x, end_y));
    }

    extended_boxes
}

#[pyfunction]
fn make_box_from_name_native(
    image: PyReadonlyArray3<'_, u8>,
    name: &str,
    padding: Option<i32>,
) -> BoxTuple {
    let padding = padding.unwrap_or(2);
    let shape = image.as_array().shape().to_vec();
    let image_height = shape.first().copied().unwrap_or(0) as i32;
    let image_width = shape.get(1).copied().unwrap_or(0) as i32;
    let text_w = (name.chars().count() as i32) * 20;
    let text_h = 22;

    let start_x = (text_w - padding).max(0);
    let start_y = (text_h - padding).max(0);
    let end_x = (text_w + padding).min(image_width);
    let end_y = (text_h + padding).min(image_height);
    (start_x, start_y, end_x, end_y)
}

#[pyfunction]
fn is_gibberish(text: &str) -> bool {
    is_gibberish_inner(text)
}

#[pyfunction]
fn gibberish_score(text: &str) -> f64 {
    gibberish_score_inner(text)
}

#[pyfunction]
fn looks_structured_overlay_text(text: &str) -> bool {
    looks_structured_inner(text)
}

#[pyfunction]
fn normalize_ocr_text(text: &str) -> String {
    normalize_text_inner(text)
}

#[pyfunction]
fn candidate_rank(text: &str, conf: f64) -> (u8, u8, f64, f64, isize) {
    let is_empty = if text.is_empty() { 1 } else { 0 };
    let is_gib = if !text.is_empty() && is_gibberish_inner(text) {
        1
    } else {
        0
    };
    let gib_score = gibberish_score_inner(text);
    (
        is_empty,
        is_gib,
        gib_score,
        -conf,
        -(text.chars().count() as isize),
    )
}

#[pyfunction]
fn fuzzy_match_best<'py>(
    py: Python<'py>,
    snippet_text: &str,
    candidates: &Bound<'py, PyList>,
    threshold: Option<f64>,
) -> PyResult<(Option<String>, f64)> {
    let threshold = threshold.unwrap_or(0.7);
    let candidate_values: Vec<String> = candidates
        .iter()
        .map(|item| item.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;

    let snippet = snippet_text.to_owned();
    let best = py.allow_threads(move || {
        candidate_values
            .par_iter()
            .map(|candidate| {
                let ratio = jaro_winkler(&snippet, candidate);
                (candidate.clone(), ratio)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
    });

    match best {
        Some((best_match, best_ratio)) if best_ratio >= threshold => {
            Ok((Some(best_match), best_ratio))
        }
        Some((_, best_ratio)) => Ok((None, best_ratio)),
        None => Ok((None, 0.0)),
    }
}

#[pymodule]
fn _lx_anonymizer_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filter_empty_boxes_native, m)?)?;
    m.add_function(wrap_pyfunction!(combine_boxes_native, m)?)?;
    m.add_function(wrap_pyfunction!(close_to_box_native, m)?)?;
    m.add_function(wrap_pyfunction!(make_box_from_device_list_native, m)?)?;
    m.add_function(wrap_pyfunction!(find_or_create_close_box_native, m)?)?;
    m.add_function(wrap_pyfunction!(get_dominant_color_native, m)?)?;
    m.add_function(wrap_pyfunction!(extend_boxes_if_needed_native, m)?)?;
    m.add_function(wrap_pyfunction!(make_box_from_name_native, m)?)?;
    m.add_function(wrap_pyfunction!(is_gibberish, m)?)?;
    m.add_function(wrap_pyfunction!(gibberish_score, m)?)?;
    m.add_function(wrap_pyfunction!(looks_structured_overlay_text, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_ocr_text, m)?)?;
    m.add_function(wrap_pyfunction!(candidate_rank, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy_match_best, m)?)?;
    Ok(())
}
