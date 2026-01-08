from typing import Optional
from deepeval.models import GPTModel


class CustomGPTModel(GPTModel):
    """
    Custom OpenAI model connector for LiteLLM proxy.
    Supports both standard and provider/model formats (e.g., 'azure/gpt-4o-mini').
    """

    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=None,
            _openai_api_key=_openai_api_key,
            base_url=base_url,
            temperature=temperature,
            *args,
            **kwargs,
        )
        self.model_name = model

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on input and output tokens. Returns 0.0 for non-GPT models."""
        return 0.0
