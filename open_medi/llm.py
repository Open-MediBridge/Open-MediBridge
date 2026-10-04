"""Thin client for any OpenAI-compatible chat completions endpoint.

Defaults to a local Ollama server; point LLM_BASE_URL/LLM_API_KEY/LLM_MODEL
at OpenAI, Groq, Z.ai, vLLM, etc. without code changes.
"""

from openai import OpenAI

from open_medi.config import Settings, get_settings


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot be reached or returns nothing."""


class LLM:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                timeout=120.0,
            )
        return self._client

    def complete(self, system: str, user: str) -> str:
        """One chat completion, requesting strict JSON with a plain fallback."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Several OpenAI-compatible servers (older Ollama, vLLM builds, ...)
            # reject response_format; retry without it and parse leniently.
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    temperature=self.settings.llm_temperature,
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001 - surface a single clear error
                raise LLMError(
                    f"LLM request failed: {exc}. Check that your backend is "
                    "running and LLM_BASE_URL / LLM_API_KEY / LLM_MODEL are "
                    "correct (see .env.example)."
                ) from exc
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMError("LLM returned an empty response.")
        return content

    def reachable(self) -> bool:
        """Cheap probe used by /api/health."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
