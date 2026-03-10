# app/providers/openai_provider.py
from __future__ import annotations
import os

try:
    from app.core.config import DEFAULT_OPENAI_MODELS, OPENAI_SYSTEM_PROMPT
except ModuleNotFoundError:
    from app.core.config import DEFAULT_OPENAI_MODELS, OPENAI_SYSTEM_PROMPT

from app.providers.base import MathProvider, ProviderResponse


class OpenAIProvider(MathProvider):
    name = "openai"

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate OpenAI API key format."""
        if not api_key or not isinstance(api_key, str):
            return False
        # OpenAI API keys start with "sk-" and are typically 51 characters
        return api_key.startswith("sk-") and len(api_key) >= 20 and len(api_key) <= 200

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        self._ready = bool(key) and self.validate_api_key(key)
        self._client = None
        self._tested = False
        if self._ready:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=key)
            except Exception:
                self._ready = False

    def test_connection(self) -> bool:
        """Test the API key by making a lightweight API call."""
        if not self._ready or self._client is None:
            return False
        
        if self._tested:
            return self._ready
            
        try:
            # Make a lightweight API call to test the key
            self._client.models.list()
            self._tested = True
            return True
        except Exception:
            self._ready = False
            self._tested = True
            return False

    def available(self) -> bool:
        return self.test_connection()

    @staticmethod
    def default_models() -> list[str]:
        return list(DEFAULT_OPENAI_MODELS)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0,
    ) -> ProviderResponse:
        if not self.available():
            return ProviderResponse(model=model, output="", error="OpenAI not available")

        try:
            # Handle o1 models differently - they don't support system messages or temperature
            if model.startswith("o1"):
                combined_prompt = f"{system_prompt or OPENAI_SYSTEM_PROMPT}\n\n{prompt}"
                messages = [{"role": "user", "content": combined_prompt}]
                kwargs = {"model": model, "messages": messages}
            else:
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt or OPENAI_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
                kwargs = {"model": model, "messages": messages, "temperature": temperature}
            
            resp = self._client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content.strip()
            # Strip code fences if the model used them
            if content.startswith("```"):
                content = content.strip("`").strip()
            return ProviderResponse(model=model, output=content)
        except Exception as e:
            return ProviderResponse(model=model, output="", error=str(e))
