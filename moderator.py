

import json
from groq import Groq

import config


class Moderator:
    def __init__(self, client: Groq):
        self.client = client
        with open(config.MODERATOR_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt = f.read()

    def moderate(self, question: str) -> dict:
        """
        Renvoie un dict {"is_prompt_injection": bool}.
        En cas d'erreur de parsing JSON, on choisit la position prudente
        (on considère la question comme suspecte) plutôt que de planter.
        """
        response = self.client.chat.completions.create(
            model=config.MODERATION_MODEL_NAME,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw = response.choices[0].message.content

        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[Moderator] Réponse non-JSON reçue, on bloque par prudence : {raw!r}")
            return {"is_prompt_injection": True}

        # Garde-fou : si la clé attendue est absente, on bloque aussi par prudence.
        if "is_prompt_injection" not in decision:
            return {"is_prompt_injection": True}

        return decision
