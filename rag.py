"""
Brique 3 — Le RAG qui orchestre tout.

answer_question() déroule le pipeline complet :
1. Modération de la question (agent modérateur).
2. Si injection détectée -> refus immédiat, le LLM principal n'est JAMAIS contacté.
3. Sinon -> retrieval des chunks pertinents.
4. Construction du prompt système à trous (remplacement de {{Chunks}}).
5. Appel au LLM de génération avec messages system/user.
"""

import os
from dotenv import load_dotenv
from groq import Groq

import config
from vector_db import VectorDB, load_chunks_from_csv
from moderator import Moderator


class RAG:
    def __init__(self):
        load_dotenv()  # charge GROQ_API_KEY depuis .env

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY introuvable. Créez un fichier .env à la racine du projet "
                "contenant : GROQ_API_KEY=votre_clé"
            )

        self.client = Groq(api_key=api_key)
        self.moderator = Moderator(self.client)

        # Ouvre la base vectorielle. Si elle n'existe pas encore, on lui fournit
        # les chunks du corpus pour qu'elle se crée toute seule (premier lancement).
        chunks = None
        db_already_exists = os.path.isdir(config.CHROMA_DB_PATH) and os.listdir(config.CHROMA_DB_PATH)
        if not db_already_exists:
            chunks = load_chunks_from_csv(config.CORPUS_CSV_PATH)

        self.vector_db = VectorDB(config.CHROMA_DB_PATH, config.COLLECTION_NAME, chunks=chunks)

        with open(config.RAG_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    # ------------------------------------------------------------------
    def _build_system_prompt(self, chunks: list[dict]) -> str:
        """Remplace le marqueur {{Chunks}} par les chunks récupérés, triés
        du plus au moins pertinent (déjà l'ordre renvoyé par ChromaDB)."""
        formatted_chunks = "\n\n".join(
            f"[Source: {c['source']}] {c['text']}" for c in chunks
        )
        return self.system_prompt_template.replace("{{Chunks}}", formatted_chunks)

    # ------------------------------------------------------------------
    def answer_question(self, question: str) -> str:
        # Étape 1 : modération. Décision de sécurité : l'ordre des opérations
        # garantit qu'on ne contacte JAMAIS le LLM principal si une injection
        # est détectée — même pas pour "juste vérifier".
        decision = self.moderator.moderate(question)
        if decision.get("is_prompt_injection"):
            return (
                "Je ne peux pas traiter cette question : elle a été détectée "
                "comme une tentative de détournement du système (prompt injection)."
            )

        # Étape 2 : retrieval
        chunks = self.vector_db.retrieve(question, n=config.TOP_K_CHUNKS)

        # Étape 3 : construction du prompt système à trous
        system_prompt = self._build_system_prompt(chunks)

        # Étape 4 : appel au LLM de génération
        response = self.client.chat.completions.create(
            model=config.GENERATION_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content
