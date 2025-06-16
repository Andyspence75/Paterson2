import spacy
import subprocess
import sys

def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            return spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError(f"Could not load or download spaCy model: {e}")

nlp = load_spacy_model()

def redact_text(text):
    doc = nlp(text)
    redacted = []
    for token in doc:
        if token.ent_type_ in ["PERSON", "GPE", "ORG", "LOC"]:
            redacted.append("[REDACTED]")
        else:
            redacted.append(token.text)
    return " ".join(redacted)