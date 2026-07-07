"""
Configuration du projet RAG
"""

# Modèle d'embedding utilisé pour transformer les textes en vecteurs
EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"

# Nom de la collection ChromaDB
COLLECTION_NAME = "mini_rag"

# Dossier où la base vectorielle sera enregistrée
CHROMA_DB_PATH = "./db"

# Nombre de documents à récupérer lors d'une recherche
TOP_K = 3