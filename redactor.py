
import spacy
from langchain.schema import Document

nlp = spacy.load("en_core_web_sm")

def redact_text(docs):
    redacted_docs = []
    for doc in docs:
        text = doc.page_content
        nlp_doc = nlp(text)
        redacted_text = text
        for ent in reversed(nlp_doc.ents):
            if ent.label_ in ("PERSON", "GPE", "ORG", "LOC"):
                redacted_text = redacted_text[:ent.start_char] + "[REDACTED]" + redacted_text[ent.end_char:]
        redacted_docs.append(Document(page_content=redacted_text, metadata=doc.metadata))
    return redacted_docs
