"""Utility functions to manipulate JSON-LD files with python built-in structures"""

from pathlib import Path
import json


def load_jsonld(file_path: Path) -> dict:
    """Utility function to load a JSON-LD file

    Args:
        path (Path): File path

    Returns:
        dict: deserialized json dictionary
    """
    with open(file_path, encoding="utf-8") as f:
        try:
            return json.load(f)["@graph"]
        except KeyError as e:
            raise KeyError(f"JSON-LD with wrong structur: {e}")


def extract_taxonomy_ids(doc: dict) -> tuple[str, set[str]]:
    """Utilify function to obtain all the taxonomy 'about' ids that has a document (working paper) and
    the preferred language of the working paper

    Args:
        doc (dict): metadata of a working paper

    Returns:
        tuple[str, set[str]]: tuple that contains the language code and a list of all the 'about' ids related to the document.
    """
    about = doc.get("schema:about", [])
    lang = doc.get("schema:inLanguage").get("@value")
    return lang, {term["@id"] for term in about if "@id" in term}


def value_lang(labels: list[dict[str, str]] | dict[str, str], lang: str) -> str:
    """Utility function to obtain the preferred language description from the taxonomy

    Args:
        labels (list[dict[str]]): list of possible descriptions in all available languages
        lang (str): preferred language

    Returns:
        str: Description in the preferred language
    """
    if isinstance(labels, list):
        for label in labels:
            if label["@language"] == lang:
                return label["@value"]
    elif isinstance(labels, dict):
        if labels["@language"] == lang:
            return labels["@value"]


def obtain_labels(tax_docs: list[dict], doc: dict) -> set[str]:
    """Utility function that queries into the taxonomy and returns a set of all the topics/authors related to
    the "aboutness" of a document

    Args:
        tax_docs (list[dict]): list of taxonomy details
        doc (dict): metadata of a working paper

    Returns:
        set[str]: set of all the descriptions related to the "aboutness" of a document.
    """
    final_set = set()

    lang, tax_ids = extract_taxonomy_ids(doc)

    for tax in tax_docs:
        if tax["@id"] in tax_ids:
            final_set.add(value_lang(tax["skos:prefLabel"], lang))

    return final_set

def get_taxonomy_by_lang(tax_docs: list[dict], lang: str) -> set[str]:
    """Get taxonomy topics that are in the IdBTopics schemas http://thesaurus.iadb.org/idbthesauri/IdBTopics
    """
    idb_topics = 'http://thesaurus.iadb.org/idbthesauri/IdBTopics'

    final_set = set()
    for tax in tax_docs:
        schemes = tax.get('skos:inScheme')
        if not schemes:
            continue
        if isinstance(schemes, dict):
            schemes = [schemes]

        tax_scheme = set(item["@id"] for item in schemes if "@id" in item)
        
        if idb_topics in tax_scheme:
            final_set.add(value_lang(tax["skos:prefLabel"], lang))

    return final_set

def unique_taxonomy_schemes(tax_docs: list[dict]) -> set[str]:
    scheme = set()
    for tax in tax_docs:
        value = tax.get('skos:inScheme')
        if not value:
            continue

        if isinstance(value, dict):
            value = [value]

        ids = [item["@id"] for item in value if "@id" in item]
        scheme.update(ids)
    
    return scheme