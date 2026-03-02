import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings, Collection, QueryResult
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

class TFIDFEmbedd(EmbeddingFunction[Documents]):
    """Embedding for TF-IDF embeddings
    """
    def __init__(self, all_documents: list[str]):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(all_documents)

    def __call__(self, input: Documents) -> Embeddings:
        return self.vectorizer.transform(input).toarray()

class ChromaClient:
    """ChromaDB Client to compute text similarity analysis (semantic search)
    """
    def __init__(self, db_path: Path):
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings()
        )
        self.collections: dict[str, Collection] = {}

    def create_collection(self, collection_name: str, embedding_fn: EmbeddingFunction):
        """Creates a collection in the ChromaDB

        Args:
            collection_name (str): name of the collection
            embedding_fn (EmbeddingFunction): embedding function for this collection
        """
        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.collections[collection_name] = collection

    def add_documents(self, collection_name: str, id_list: list, texts:list, metadata: list|None = None, batch_size=100):
        """Adds documents in batch to the specified collection and keeps tracks of document's metadata

        Args:
            collection_name (str): name of the collection
            id_list (list): documents' id
            texts (list): documents' text
            metadata (list | None, optional): Documents' metadata. Defaults to None.
            batch_size (int, optional): batch size. Defaults to 100.
        """
        collection = self.collections[collection_name]

        for i in range(0, len(texts), batch_size):
            collection.add(
                documents=texts[i:i+batch_size],
                ids=id_list[i:i+batch_size],
                metadatas=metadata[i:i+batch_size] if metadata else None,
            )

    def query(self, collection_name: str, query: str, n_result: int = 5) -> QueryResult:
        """Query the ChromaDB into the specified collection and returns the top 5 
        similar documents related to the query

        Args:
            collection_name (str): name of the collection
            query (str): query from the user
            n_result (int, optional): number of results. Defaults to 5.

        Returns:
            QueryResult: Top-k nearest neighbor records in the collection
        """
        collection = self.collections[collection_name] 
        return collection.query( query_texts=[query], n_results=n_result)

    def _doc_hit_at_k(self, collection_name: str, query: str, true_doc_id: str, k: int = 5) -> float:

        results = self.query(collection_name, query, n_result=k)
        retrieved_ids = results.get("ids", [[]])[0]
        return float(true_doc_id in set(retrieved_ids))

    def corpus_accuracy_at_k(self, collection_name: str, docs: list[dict], k: int = 5) -> float:
        
        accs = list(self.get_docs_accuracy(collection_name, docs, k).values())

        return float(np.mean(accs)) if accs else 0.0
    
    def get_docs_accuracy(self, collection_name: str, docs: list[dict], k: int = 5):
        accs = {}
        for doc in docs:
            doc_id = doc["id"]
            about_terms = doc.get("about", [])
            if not about_terms:
                continue

            compound_about = " ".join(about_terms)    
            
            accs[doc_id] = self._doc_hit_at_k(collection_name, compound_about, doc_id, k)

        return accs
    
    def export_accs(self, collection_name: str, docs: list[dict], k: int = 5):
        
        accs = self.get_docs_accuracy(collection_name, docs, k)
        with open(DATA / f"accs_{collection_name}.json", "w") as file:
            json.dump(accs,file)