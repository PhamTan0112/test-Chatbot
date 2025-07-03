from difflib import get_close_matches
import json

with open("symptom_list.json", encoding="utf-8") as f:
    STANDARD_SYMPTOMS = json.load(f)

def normalize_symptom(symptom: str) -> str:
    clean_sym = symptom.strip().lower()
    match = get_close_matches(clean_sym, STANDARD_SYMPTOMS, n=1, cutoff=0.6)
    return match[0] if match else clean_sym