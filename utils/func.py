import re

def extract_answer_and_confidence(input_string):
    pattern = r"Answer and Confidence \(0-100\): (\d+), (\d+)%"
    match = re.search(pattern, input_string)
    
    if match:
        answer = int(match.group(1))
        confidence = int(match.group(2))
        if answer > 4 or answer == 0: return 0, 0
        return answer, confidence
    else:
        return 0, 0

def scaled_probe(val, min_val, max_val):
    return int(((val - min_val) / (max_val - min_val)) * 50 + 50)