"""OpenAI-compatible chat client for the LLM agent.

Credentials come from environment variables - never hardcode keys:

    GUANDAN_LLM_API_KEY   - API key (required)
    GUANDAN_LLM_BASE_URL  - endpoint, e.g. https://api.deepseek.com/
                            (optional; defaults to the official OpenAI API)
    GUANDAN_LLM_MODEL     - model name, e.g. deepseek-chat (required
                            unless passed explicitly)

Any OpenAI-compatible endpoint works (OpenAI, DeepSeek, DashScope,
SiliconFlow, vLLM/Ollama serving, ...).
"""

import logging
import os
import re

logger = logging.getLogger('guandan.llm')

DEFAULT_SYSTEM_PROMPT = (
    '你是一名专业的中国掼蛋扑克牌手，你擅长分析场上的局势打出最优的牌。'
)


def strip_think_blocks(text):
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat completions endpoint."""

    def __init__(self, model=None, base_url=None, api_key=None,
                 temperature=0.5, max_tokens=8192, max_retries=2):
        from openai import OpenAI  # local import keeps openai optional

        api_key = api_key or os.environ.get('GUANDAN_LLM_API_KEY')
        if not api_key:
            raise RuntimeError(
                'Set GUANDAN_LLM_API_KEY (and optionally '
                'GUANDAN_LLM_BASE_URL / GUANDAN_LLM_MODEL) before using '
                'the LLM agent.')
        base_url = base_url or os.environ.get('GUANDAN_LLM_BASE_URL')
        self.model = model or os.environ.get('GUANDAN_LLM_MODEL')
        if not self.model:
            raise RuntimeError('No model given: pass model=... or set '
                               'GUANDAN_LLM_MODEL.')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
        """Return the model's reply text for a single-turn prompt.

        Retries on transport errors; returns None when all attempts fail
        so the caller can fall back to a non-LLM policy.
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        for attempt in range(1 + self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content or ''
                return strip_think_blocks(content)
            except Exception as error:  # network/API errors: retry
                logger.warning('LLM call failed (attempt %s/%s): %s',
                               attempt + 1, 1 + self.max_retries, error)
        return None
