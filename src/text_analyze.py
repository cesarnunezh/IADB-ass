
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import obtain_labels

def _get_value(obj: list | dict):
    if isinstance(obj, list):
        return obj[0]['@value']
    elif isinstance(obj, dict):
        return obj['@value']    

def get_corpus_by_lang(docs: list[dict], tax_docs: list[dict]) -> dict[dict]:
    """Function that transform a JSON-LD deserialized dictionary and 
    extract the corpus of description and full text.

    Args:
        docs (list[dict]): _description_

    Returns:
        dict[dict]: _description_
    """

    corpus: dict[str, dict[str, str]] = {}

    for doc in docs:

        doc_id = doc['@id']
        lang = doc['schema:inLanguage']['@value']
        desc = doc['schema:description']
        text = doc['schema:text']
        keywords = doc.get('schema:keywords', "")

        corpus[doc_id] = {
            "lang" : lang,
            "about": obtain_labels(tax_docs, doc),
            "desc" : _get_value(desc),
            "text" : _get_value(text),
            "keywords" : _get_value(keywords)

        }

    return corpus
