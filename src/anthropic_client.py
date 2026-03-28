import time
from typing import Dict, List

from anthropic import Anthropic


class AnthropicChatClient:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        retries: int = 5,
    ) -> str:
        #Anthropic takes system prompt as a separate parameter, not in messages.
        #Split it out from the messages list.
        system_text = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                user_messages.append(m)

        delay = 1.0
        last_err = None
        for _ in range(retries):
            try:
                resp = self.client.messages.create(
                    model=model,
                    system=system_text,
                    messages=user_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_output_tokens,
                )
                #Anthropic returns content as a list of content blocks
                return resp.content[0].text if resp.content else ""
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
        raise RuntimeError(f"Anthropic request failed after retries: {last_err}")