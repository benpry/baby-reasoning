from __future__ import annotations
import requests
from baby_reasoning.tasks.base import ModelBackend, ModelResponse


class VLLMBackend(ModelBackend):
    """ModelBackend implementation over the vLLM OpenAI-compatible API."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000") -> None:
        self._model = model
        self.base_url = base_url.rstrip("/")

    @property
    def model(self) -> str:
        return self._model

    def _post(self, payload: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str) -> ModelResponse:
        data = self._post(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 50,
                "logprobs": 1,
            }
        )
        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs")
        token_logprobs = (
            logprobs_data.get("token_logprobs")
            if isinstance(logprobs_data, dict)
            else None
        )
        return ModelResponse(
            text=choice.get("text", "").rstrip(),
            token_logprobs=token_logprobs,
        )

    def score_completion(self, prompt: str, completion: str) -> float | None:
        """Return sum of token log probs for the completion only, or None if unsupported.

        Uses text_offset from the echoed response to identify which tokens
        belong to the completion (offset >= len(prompt)) and sums only those.
        """
        data = self._post(
            {
                "model": self.model,
                "prompt": prompt + completion,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
            }
        )
        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs")
        if not isinstance(logprobs_data, dict):
            return None
        token_logprobs = logprobs_data.get("token_logprobs")
        if token_logprobs is None:
            return None
        text_offset = logprobs_data.get("text_offset")
        if text_offset is None:
            return None
        prompt_len = len(prompt)
        return sum(
            lp for lp, off in zip(token_logprobs, text_offset)
            if off >= prompt_len and lp is not None
        )
