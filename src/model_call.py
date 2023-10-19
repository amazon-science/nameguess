import os
from typing import Any
from abc import ABC, abstractmethod
import openai


class BaseLLM(ABC):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        print(
            f"======================= Now you are calling {model_name} ======================= "
        )

    @abstractmethod
    def prepare_model(self):
        pass


class OpenaiLLM(BaseLLM):
    def __init__(
        self, model_name: str, openai_key: str = None, openai_api_key_path: str = None
    ) -> None:
        super().__init__(model_name)
        self.openai_key = openai_key
        self.openai_api_key_path = openai_api_key_path
        self.prepare_model(self.openai_key, self.openai_api_key_path)

    def prepare_model(self, openai_key: str, openai_api_key_path: str) -> None:
        if openai_key is None:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key is None:
                assert openai_api_key_path is not None
                # read OpenAI key if needed
                with open(self.open_api_key_path, "r") as f:
                    openai_key = f.read().strip("\n")
        openai.api_key = openai_key

    def __call__(
        self,
        query,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if self.model_name == "gpt-3.5-turbo" or self.model_name == 'gpt-4':
            completion = openai.ChatCompletion.create(
                model=self.model_name,
                temperature=temperature,
                messages=[{"role": "user", "content": query}],
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content

        if self.model_name == "text-davinci-003":
            completion = openai.Completion.create(
                model=self.model_name,
                temperature=temperature,
                prompt=query,
                max_tokens=max_tokens,
            )
            return completion.choices[0].text
