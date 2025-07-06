import spacy


class NER:
    """Extremely fast statistical entity recognition system
    that assigns labels to contiguous spans of tokens. The default
    trained pipelines can identify a variety of named and numeric
    entities, including companies, locations, organizations and
    products. Feedback updates the model with new known entities
    with score (eventually). Score of 0 gets labelled trash
    in the Gmail inbox.
    """

    def __init__(self):
        self.rag = None
        self.message = None

    def get_entities(self) -> None:
        # Access for user's known entities from previous feedback
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)
