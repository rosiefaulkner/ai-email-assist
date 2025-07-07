import spacy
from spacy.tokens import Doc, Token
from typing import List, Tuple

# Load the English language model
nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract entities from text with custom product recognition."""
    doc = nlp(text)
    entities = []

    for token in doc:
        if token.like_email:
            email = token.text
            domain_part = email.split("@")[1]
            domain = domain_part.split(".")[0]
            entities.append((email, domain))
        elif token.like_url:
            url = token.text
            entities.append((url, "URL"))
        elif token.ent_type_:
            entities.append((token.text, token.ent_type_))
        else:
            if token.pos_ not in [
                "PROPN",
                "NOUN",
                "PUNCT",
                "SYM",
                "PRON",
                "VERB",
                "ADJ",
                "ADV",
                "NUM",
                "SPACE",
                "DET",
                "CCONJ",
                "ADP",
                "PART",
                "INTJ",
                "X",
                "AUX",
                "SCONJ",
            ]:
                entities.append((token.text, token.pos_))

    return entities

