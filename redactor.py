
import spacy
import subprocess
import sys

# Load spaCy model or download it
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def redact_text(text):
    doc = nlp(text)
    redacted = []
    for token in doc:
        if token.ent_type_ in ["PERSON", "GPE", "ORG", "LOC"]:
            redacted.append("[REDACTED]")
        else:
            redacted.append(token.text)
    return " ".join(redacted)
