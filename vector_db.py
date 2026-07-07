import csv
import os

import chromadb
from sentence_transformers import SentenceTransformer

import config


def load_chunks_from_csv(csv_path: str) -> list[dict]:
    """Charge le corpus depuis le CSV (colonnes attendues : id, text, source).

    Renvoie une liste de dicts {"id":..., "text":..., "source":...}.
    """
    chunks = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "source": row["source"],
                }
            )
    return chunks


class VectorDB:
    def __init__(self, db_path: str, collection_name: str, chunks: list[dict] | None = None):
        self.client = chromadb.PersistentClient(path=db_path)

        try:
            self.collection = self.client.get_collection(collection_name)
            self._reload(collection_name)

        except Exception:

            if chunks:
                self._create(collection_name, chunks)
            else:
                raise ValueError(
                    "Aucune base trouvée et aucun corpus fourni."
                )

        if db_already_exists:
            self._reload(collection_name)
        elif chunks:
            self._create(collection_name, chunks)
        else:
            raise ValueError(
                f"Aucune base trouvée à '{db_path}' et aucun chunk fourni pour en créer une. "
                "Fournissez des chunks (via load_chunks_from_csv) au premier lancement."
            )

    # ------------------------------------------------------------------
    def _create(self, collection_name: str, chunks: list[dict]) -> None:
        """Crée la collection, encode le corpus, insère tout, et grave le
        nom du modèle d'embedding dans les métadonnées de la collection."""
        embedding_model_name = config.EMBEDDING_MODEL_NAME
        self.model = SentenceTransformer(embedding_model_name)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": embedding_model_name},
        )

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,  # cosinus <=> produit scalaire sur vecteurs normalisés
            show_progress_bar=True,
        )

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[{"source": c["source"]} for c in chunks],
        )

    # ------------------------------------------------------------------
    def _reload(self, collection_name: str) -> None:
        """Recharge une base existante : aucun encodage de corpus ici.
        Le modèle chargé est celui écrit dans les métadonnées de la
        collection au moment de sa création — pas celui de config.py."""
        self.collection = self.client.get_or_create_collection(name=collection_name)

        embedding_model_name = self.collection.metadata.get("embedding_model")
        if not embedding_model_name:
            # Collection créée sans cette métadonnée (ancienne version, ou
            # créée hors de ce code) : on retombe sur la config par défaut.
            embedding_model_name = config.EMBEDDING_MODEL_NAME

        self.model = SentenceTransformer(embedding_model_name)

    # ------------------------------------------------------------------
    def _encode_query(self, question: str):
        return self.model.encode(
            [question],
            normalize_embeddings=True,
        ).tolist()

    # ------------------------------------------------------------------
    def retrieve(self, question: str, n: int = 3) -> list[dict]:
        """Encode la question et renvoie les n chunks les plus proches.

        Respecte le contrat convenu avec la partie de Rime :
            [{"text": ..., "source": ..., "distance": float}, ...]
        triés du plus au moins pertinent (déjà l'ordre renvoyé par Chroma).
        """
        query_embedding = self._encode_query(question)

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n,
        )

        chunks = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            chunks.append(
                {
                    "text": text,
                    "source": metadata.get("source", "inconnue"),
                    "distance": distance,
                }
            )

        return chunks


# ----------------------------------------------------------------------
# Test manuel : lance `python vector_db.py` pour vérifier la brique seule,
# comme demandé section 3.3 du TP (5 questions, vérifier que les bons
# chunks remontent en tête avant de brancher quoi que ce soit d'autre).
if __name__ == "__main__":
    db_already_exists = os.path.isdir(config.CHROMA_DB_PATH) and os.listdir(config.CHROMA_DB_PATH)

    chunks = None
    if not db_already_exists:
        chunks = load_chunks_from_csv(config.CORPUS_CSV_PATH)
        print(f"{len(chunks)} chunks chargés depuis {config.CORPUS_CSV_PATH}, création de la base...")
    else:
        print("Base existante détectée, rechargement...")

    db = VectorDB(config.CHROMA_DB_PATH, config.COLLECTION_NAME, chunks=chunks)

    questions_test = [
        "Quelle est la couleur du chat de Bob ?",
        "Comment s'appelle le chien d'Alice ?",
        "Le chien d'Alice a-t-il peur de quelque chose ?",
        "Quel jour le chat de Bob miaule-t-il ?",
        "Où dort Henri le chat ?",
    ]

    for q in questions_test:
        print(f"\nQuestion : {q}")
        for chunk in db.retrieve(q, n=3):
            print(f"  [{chunk['distance']:.4f}] ({chunk['source']}) {chunk['text']}")