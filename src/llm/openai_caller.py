"""OpenAI API caller — drop-in replacement for VLLMCaller.

Used for the judge step so we can call GPT-5.4-mini (or any OpenAI model)
instead of running a local vLLM instance for LLM-as-Judge scoring.
"""

import os
import concurrent.futures
from openai import OpenAI


class OpenAICallerConfig:
    def __init__(self, model_name, temperature=0.2, max_tokens=2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens


class OpenAICaller:
    def __init__(self, config: OpenAICallerConfig):
        self.config = config
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        print(f"OpenAICaller ready. model={config.model_name}, "
              f"temperature={config.temperature}")

    def call(self, messages: list[dict]) -> str:
        """Single conversation. Matches VLLMCaller.call interface."""
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content

    def call_batch(self, samples: list[list[dict]]) -> list[str]:
        """Concurrent API calls. Matches VLLMCaller.call_batch interface."""
        max_workers = min(len(samples), 16)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self.call, conv) for conv in samples]
            return [f.result() for f in futures]
