"""
Moteur RAG (Retrieval-Augmented Generation) pour le catalogue produits.
Indexe dim_product.csv dans une base de données vectorielle ChromaDB
et permet la recherche sémantique basée sur les caractéristiques des produits.
"""

import os
import pandas as pd
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")


class ProductRAGEngine:

    def __init__(self, csv_path: str = None):
        self.csv_path = csv_path or os.path.join(DATA_DIR, "dim_product.csv")
        self.vector_store = None
        self._init_rag()

    def _init_rag(self):
        """Initialise la base vectorielle ChromaDB avec le fichier dim_product.csv."""
        if not os.path.exists(self.csv_path):
            print(f"RAG: Fichier {self.csv_path} introuvable.")
            return

        df = pd.read_csv(self.csv_path)

        documents = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            content_parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            doc_text = " | ".join(content_parts)
            documents.append(doc_text)
            metadatas.append({"row_id": int(idx)})
            ids.append(f"prod_{idx}")

        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            collection = client.get_or_create_collection(name="product_catalog")

            # Remplit la collection si vide
            if collection.count() == 0 and documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            self.vector_store = collection
            print(f"RAG: ChromaDB initialisé avec {collection.count()} produits.")
        except Exception as e:
            print(f"RAG ChromaDB Note: {e} (Fallback pandas sémantique disponible)")
            self.df = df

    def query(self, search_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Recherche sémantique dans le catalogue."""
        if self.vector_store:
            try:
                results = self.vector_store.query(
                    query_texts=[search_text],
                    n_results=k
                )
                output = []
                if results and "documents" in results and results["documents"]:
                    for doc in results["documents"][0]:
                        output.append({"content": doc})
                return output
            except Exception as e:
                print(f"Erreur recherche ChromaDB: {e}")

        # Fallback si Chroma non disponible
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            matches = df[
                df.apply(
                    lambda r: r.astype(str).str.contains(search_text, case=False).any(),
                    axis=1
                )
            ].head(k)
            return [{"content": str(dict(r))} for _, r in matches.iterrows()]

        return []


_rag_instance = None


def get_rag_engine() -> ProductRAGEngine:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = ProductRAGEngine()
    return _rag_instance
