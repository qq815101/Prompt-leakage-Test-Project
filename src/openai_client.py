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
        token_param = "max_completion_tokens"
        for attempt in range(retries):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    token_param: max_output_tokens,
                }
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e)
                # If the API rejects the token param name, swap and retry immediately
                if "max_completion_tokens" in err_str and token_param == "max_completion_tokens":
                    token_param = "max_tokens"
                    continue
                if "max_tokens" in err_str and token_param == "max_tokens":
                    token_param = "max_completion_tokens"
                    continue
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
        raise RuntimeError(f"OpenAI request failed after retries: {last_err}")