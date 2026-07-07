"""
Configuration centrale du projet.

⚠️ Ce fichier est PARTAGÉ avec Fatima. Chacune en a besoin (elle pour VectorDB,
toi pour Moderator/RAG). Mettez-vous d'accord à deux avant de le figer, puis
gardez UNE SEULE version dans le dépôt Git — ne laissez pas deux config.py
différents traîner, sinon vous retomberez exactement dans le bug silencieux
décrit plus bas (modèle différent entre indexation et interrogation).
"""

# --- Modèle d'embedding (utilisé par VectorDB — géré par Fatima) ---
EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"

# --- Modèle de génération (utilisé par RAG — TA partie) ---
GENERATION_MODEL_NAME = "llama-3.3-70b-versatile"

# --- Modèle de modération (utilisé par Moderator — TA partie) ---
# Vérifie le nom exact disponible dans le catalogue "safeguard" sur console.groq.com
MODERATION_MODEL_NAME = "meta-llama/llama-guard-4-12b"

# --- Chemins (VectorDB — géré par Fatima) ---
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "corpus_rag"
CORPUS_CSV_PATH = "data/corpus.csv"

# --- Prompts (TA partie) ---
RAG_SYSTEM_PROMPT_PATH = "prompts/system_rag.txt"
MODERATOR_SYSTEM_PROMPT_PATH = "prompts/system_moderator.txt"

# --- Paramètres de retrieval (utilisé par TA partie, défini ici pour rester au même endroit) ---
TOP_K_CHUNKS = 3
