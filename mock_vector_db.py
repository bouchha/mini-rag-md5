"""
BOUCHON DE TEST — pas la vraie implémentation.

Ceci n'est PAS le fichier vector_db.py final (celui-là, c'est la partie de Fatima,
avec la vraie base ChromaDB persistée). C'est un faux VectorDB qui renvoie des
chunks bidons, pour que tu puisses développer et tester rag.py et moderator.py
SANS attendre que Fatima ait fini sa partie.

Interface à respecter (le contrat entre vos deux parties) :

    db = VectorDB(chroma_db_path, collection_name, chunks=...)
    db.retrieve(question: str, n: int) -> list[dict]

    Chaque dict de la liste retournée doit avoir EXACTEMENT ces clés :
        {"text": "...", "source": "...", "distance": 0.123}

    - "text"     : le texte du chunk
    - "source"   : le nom du fichier source (colonne "source" du CSV)
    - "distance" : un float, plus petit = plus proche/pertinent

C'est ce contrat qu'il faut vérifier avec Fatima AVANT de coder chacune de son
côté. Si sa vraie classe respecte cette même signature et ce même format de
retour, tu pourras remplacer ce mock par son vrai vector_db.py sans changer
une ligne de rag.py.

Pour l'utiliser : dans rag.py, remplace temporairement
    from vector_db import VectorDB, load_chunks_from_csv
par
    from mock_vector_db import VectorDB, load_chunks_from_csv
le temps de tes tests, puis remets l'import normal une fois que Fatima t'a
donné son vrai fichier.
"""


class VectorDB:
    def __init__(self, db_path: str, collection_name: str, chunks=None):
        print("[MOCK VectorDB] Aucune vraie base chargée — ceci est un bouchon de test.")

    def retrieve(self, question: str, n: int = 3) -> list[dict]:
        # Quelques faux résultats plausibles, tirés du corpus, pour tester le
        # pipeline RAG (construction du prompt, appel au LLM) indépendamment
        # du vrai retrieval.
        faux_chunks = [
            {"text": "Le chat bleu de Bob s'appelle Henri.", "source": "carnet_de_bob", "distance": 0.12},
            {"text": "Henri, le chat de Bob, refuse de dormir ailleurs que sur le réfrigérateur.",
             "source": "carnet_de_bob", "distance": 0.31},
            {"text": "Contrairement à une idée répandue au village, le chat d'Henri n'a jamais été vert : il est et a toujours été bleu.",
             "source": "carnet_de_bob", "distance": 0.44},
        ]
        return faux_chunks[:n]


def load_chunks_from_csv(csv_path: str) -> list[dict]:
    # Pas utilisé par le mock, mais gardé pour que l'import ne casse pas
    # si rag.py appelle cette fonction.
    return []
