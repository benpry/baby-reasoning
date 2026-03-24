from __future__ import annotations
import requests
from baby_reasoning.tasks.base import ModelBackend, ModelResponse


class OllamaBackend(ModelBackend):
    """ModelBackend implementation over the Ollama HTTP API."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _post(self, payload: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str) -> ModelResponse:
        data = self._post({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "options": {"num_predict": 50},
        })
        logprobs_data = data.get("logprobs")
        token_logprobs = (
            logprobs_data.get("token_logprobs")
            if isinstance(logprobs_data, dict)
            else None
        )
        return ModelResponse(
            text=data.get("response", "").rstrip(),
            token_logprobs=token_logprobs,
        )

    def score_completion(self, prompt: str, completion: str) -> float | None:
        """Return sum of token log probs for the completion, or None if unsupported."""
        data = self._post({
            "model": self.model,
            "prompt": prompt + completion,
            "stream": False,
            "logprobs": True,
            "options": {"num_predict": 0},
        })
        logprobs_data = data.get("logprobs")
        if not isinstance(logprobs_data, dict):
            return None
        token_logprobs = logprobs_data.get("token_logprobs")
        if not token_logprobs:
            return None
        return sum(token_logprobs)
