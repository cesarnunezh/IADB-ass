# Technical Assessment — Metadata Accuracy & Consistency Analysis

César Núñez Huamán - cnunezh@uchicago.edu


## 1. Overview
This repository contains the analysis of metadata quality for IDB Working Papers using:
- JSON-LD datasets for taxonomy and documents descriptions.
- Metadata inconsistency detection.
- ChromaDB for semantic search using TF-IDF embeddings and a sentence-transformers model.
- Accuracy evaluation of `schema:about` tags based on the semantic search engine.

The objective is to evaluate:
 1. Metadata accuracy: Do assigned `schema:about` tags correctly identify the working paper under semantic search?
 2. Metadata consistency: Are the metadata fields complete, valid and consistent?

 ## 2. Repository Structure
 ```
 .
├── README.md                   # This file
├── analyze.ipynb               # Jupyter notebook for analysis
├── chroma_db/                  # ChromaDB persistent Client
├── data/                       # Datasets for the analysis
│   ├── taxonomy_labels.json
│   └── working_papers_metadata.json
├── pyproject.toml              # Library requirements
└── src/                        # Helper functions for analysis
    ├── embeds.py
    ├── text_analyze.py
    └── utils.py
 ```

### How to Run

### 1. Clone the Repository

```bash
git clone git@github.com:cesarnunezh/IADB-ass.git
cd IADB-ass
```

### 2. Install Dependencies
This project uses uv for dependency management.
```
uv sync
```
This will install all dependencies defined in pyproject.toml and uv.lock.

### 3. Verify Data Files
Ensure the following files exist in the `data/` directory:
- `working_papers_metadata.json`
- `taxonomy_labels.json`

These are required for both metadata consistency analysis and semantic evaluation.

### 4. Run the Analysis
Run via Jupyter Notebook
```
uvx jupyter notebook analyze.ipynb
```

## 3. Methodology

### 3.1. JSON-LD Normalization
The working papers and taxonomy dataset are in a JSON-LD format which is a new format to me. As I'm used to work with JSON files I work with these datasets as if were a JSON file and the first task was the normalization step as follow:
- Extract the content from the `@graph` key in the JSON-LD files.
- Extract taxonomy URIs from `schema:about`
- Map URIs to multilingual labels
- Normalize list/dict inconsistencies in: `skos:prefLabel` and `skos:inScheme`.

This normalization process is mainly handled by [`src/utils.py`](./src/utils.py)

### 3.2 Corpus Construction
The corpus builder extracts:
- Language
- Description
- Full text
- Cleaned about labels
- Raw keywords (which I did not used due the poor formatting -- See section 5)

This corpus construction is mainly handled by [`src/text_analyze.py`](./src/text_analyze.py)

### 3.3 Semantic Search Engine
A persistent ChromaDB instance is used to evaluate tag descriptiveness.

Embedding methods:
- TF-IDF Vectorization 
- Sentence Transformers

Similarity methods:
- Cosine similarity
- HNSW index (cosine space)

This is mainly handled by [`src/embeds.py`](./src/embeds.py)

## 4. Metadata Accuracy Definition
We evaluate whether a working paper’s assigned `schema:about` labels are sufficiently descriptive to retrieve the same document under semantic search. If the assigned tags truly describe the document, then querying the vector database using only those tags should retrieve the document itself among the top-k most similar results.

### Notation

Let:

- $ D = \{1, \dots, n\} $ be the set of documents.
- $ id_i $ be the unique identifier of document $ i $.
- $ A_i = \{a_{i1}, \dots, a_{im_i}\} $ be the set of `about` labels assigned to document $ i $.
- $ k $ be the number of retrieved documents in semantic search.

We construct a query for each document by concatenating its assigned `about` labels:

$$
q_i = \text{join}(A_i)
$$

Let $ R_k(q_i) $ denote the set of document IDs retrieved in the top-k results when querying the vector database with $ q_i $.

### Document-Level Hit

We define a binary hit indicator:

$$
h_i = \mathbf{1}\{ id_i \in R_k(q_i) \}
$$

This equals:

- **1** if the document is retrieved in the top-k results
- **0** otherwise

### Corpus-Level Accuracy

We compute corpus-level accuracy as the average hit rate across all documents with non-empty `about` labels.

Let $ D' \subseteq D $ denote the subset of documents that contain at least one `about` label.

$$
\text{Accuracy@k} = \frac{1}{|D'|} \sum_{i \in D'} h_i
$$

This metric measures how often assigned `about` tags are sufficient to uniquely identify their own document under semantic retrieval.

This is mainly handled by the methods defined in [`src/embeds.py:ChromaClient`](./src/embeds.py)


### Accuracy@k Results

The results show that transformer-based embeddings consistently outperform TF-IDF across all values of *k*, particularly at strict retrieval levels (e.g., k = 1). Accuracy increases monotonically as *k* grows, indicating that most documents can be correctly identified within a small candidate set. While TF-IDF benefits from using full text, transformer embeddings achieve strong performance even when using only document descriptions, suggesting that semantic representations better capture the “aboutness” of the working papers, rather than only term frequencies.


| k | TF-IDF (Descriptions) | TF-IDF (Full Text) | Transformer (Descriptions) | Transformer (Full Text) |
|---|------------------------|-------------------|-----------------------------|--------------------------|
| 1 | 0.5141 | 0.5423 | 0.7535 | 0.6268 |
| 3 | 0.7113 | 0.7676 | 0.9366 | 0.8944 |
| 5 | 0.8239 | 0.8732 | 0.9648 | 0.9296 |
| 7 | 0.8521 | 0.9296 | 0.9718 | 0.9577 |
| 9 | 0.8803 | 0.9507 | 0.9718 | 0.9577 |


## 5. Metadata Consistency Framework

I assessed attribute completeness across the 150 working paper records. Within 26 metadata attributes, the results indicate that while `idb:jelCode` is nearly universally applied, `schema:keywords` and `schema:spatialCoverage` are not consistently populated. This suggests partial implementation of descriptive and geographic metadata across documents. The rest of the attributes are complete in the sense that are present in all working papers' metatada.

| Attribute               | Documents Present | Coverage |
|-------------------------|------------------|----------|
| `schema:keywords`       | 130 / 150        | 86.7%    |
| `idb:jelCode`           | 149 / 150        | 99.3%    |
| `schema:spatialCoverage`| 125 / 150        | 83.3%    |

### Keyword Distribution
As one of the attributes with lack less coverage are the keywords, I analyzed them uniqueness.

- **Total keyword assignments:** 617  
- **Unique keywords:** 501  
- **Total documents:** 150  

This implies a high dispersion of keyword usage, with relatively low repetition across documents. The average number of keywords per tagged document is approximately 4.75.

The large number of unique keywords relative to total assignments suggests limited standardization and potentially inconsistent tagging practices, which may affect retrieval stability and metadata harmonization.
