FIRST_NAMES = [
    "Markus",
    "Linhel",
    "Rainer",
    "Hans",
    "Anja",
    "Dorothea",
    "Doro",
    "Angelika",
    "Sven",
    "Theodor",
    "Alexander",
    "Mandy",
    "Kathrin",
    "Florian",
    "Philip",
    "Laura"
]

LAST_NAMES = [
    "Kozielski",
    "Reiter",
    "Purrer",
    "Kudlich",
    "Brand",
    "Weich",
    "Lux",
    "Meining",
    "Hann",
    "Retzbach",
    "Hose",
    "Henniger",
    "Weich",
    "Dela Cruz",
    "Wiese",
    "Weise",
    "Sodmann"
]

PATIENT_INFO_LINE_FLAG = "Patient: "
ENDOSCOPE_INFO_LINE_FLAG = "Gerät: "
EXAMINER_INFO_LINE_FLAG = "1. Unters.:"
CUT_OFF_BELOW_LINE_FLAG = "________________"


CUT_OFF_ABOVE_LINE_FLAGS = [
    ENDOSCOPE_INFO_LINE_FLAG,
    EXAMINER_INFO_LINE_FLAG,
]

CUT_OFF_BELOW_LINE_FLAGS = [
        CUT_OFF_BELOW_LINE_FLAG
    ]

# "ukw-histology-generic-patient-info-line": "Patient: "
# "ukw-endoscopy-histology-cut-off-below-line-flag-01": "$$-3"
# "cut_off_below_line_flags": [
#   "ukw-endoscopy-histology-cut-off-below-line-flag-01"
# ]
# "cut_off_above_line_flags": [
#   "ukw-endoscopy-histology-patient-info-line"
# ]
DEFAULT_SETTINGS = {
    "locale": "de_DE",
    "first_names": FIRST_NAMES,
    "last_names": LAST_NAMES,
    "text_date_format":'%d.%m.%Y',
    "flags": {
        "patient_info_line": PATIENT_INFO_LINE_FLAG,
        "endoscope_info_line": ENDOSCOPE_INFO_LINE_FLAG,
        "examiner_info_line": EXAMINER_INFO_LINE_FLAG,
        "cut_off_below": CUT_OFF_BELOW_LINE_FLAGS,
        "cut_off_above": CUT_OFF_ABOVE_LINE_FLAGS,
    }
}