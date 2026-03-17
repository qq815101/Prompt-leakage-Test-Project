import time
from typing import Dict, List

from openai import OpenAI


class OpenAIChatClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        retries: int = 5,
    ) -> str:
        # Basic exponential backoff for transient errors / rate limits
        delay = 1.0
        last_err = None
        for _ in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_output_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
        raise RuntimeError(f"OpenAI request failed after retries: {last_err}")
