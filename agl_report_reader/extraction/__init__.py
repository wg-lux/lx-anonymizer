from ..utils import get_line_by_flag
from .examination_data import extract_examination_info
from .patient_data import extract_patient_info
from .other_data import extract_endoscope_info
from lx_anonymizer.spacy_extractor import PatientDataExtractor, ExaminerDataExtractor, EndoscopeDataExtractor


def extract_report_meta(
    text,
    patient_info_line_flag,
    endoscope_info_line_flag,
    examiner_info_line_flag,
    gender_detector = None,
    verbose = True
):
    report_meta = {}

    patient_info_line = get_line_by_flag(text, patient_info_line_flag)
    # ic(patient_info_line)
    if patient_info_line:
        patient_info = extract_patient_info(patient_info_line, gender_detector)
        # ic(patient_info)
        report_meta.update(patient_info)

    endoscope_info_line = get_line_by_flag(text, endoscope_info_line_flag)
    # ic(endoscope_info_line)
    if endoscope_info_line:
        endoscope_info = extract_endoscope_info(endoscope_info_line)
        # ic(endoscope_info)
        if endoscope_info:
            report_meta.update(endoscope_info)
        # else: 
            # ic log that no endoscope info was found
            # ic("No endoscope info found")

    examiner_info_line = get_line_by_flag(text, examiner_info_line_flag)
    # ic(examiner_info_line)
    if examiner_info_line:
        # FIXME IN CURRENT HISTO PDF PROCESSING THIS RETURNS NONE
        examiner_info = extract_examination_info(examiner_info_line)
        # ic(examiner_info)
        report_meta.update(examiner_info)

    return report_meta

