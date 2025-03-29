import re

def extract_endoscope_info(line):
    pattern = r"Gerät: ([\w\s-]+)"

    match = re.search(pattern, line)
    if match:
        endoscope = match.group(1).strip()
        return {"endoscope": endoscope}
    else:
        return None