"""
LiteLLM Custom Handlers for Chatbot Integration

This module provides a unified custom LiteLLM handler that can dynamically
route requests to different chatbot APIs based on the model name.
"""

import litellm
from litellm.llms.custom_llm import CustomLLM
from litellm.files.main import ModelResponse
import httpx
from typing import List, Dict, Any, Union
import logging
import re
from openai import OpenAI

logger = logging.getLogger(__name__)


class UnifiedChatbotLLM(CustomLLM):
    """
    A unified custom LiteLLM handler that can route requests to different
    chatbot APIs based on the model name pattern.

    Model name format: chatbot-{chatbot_name}
    Example: chatbot-edison, chatbot-edison-guardrails
    """

    def __init__(self, timeout: float = 120.0):
        """
        Initialize the unified chatbot LLM handler.

        Args:
            timeout: Request timeout in seconds (default: 30.0)
        """
        super().__init__()
        self.timeout = timeout
        # Load configurations from the config module
        from .config import CHATBOT_CONFIGS

        self.chatbot_configs = CHATBOT_CONFIGS

    def _extract_chatbot_name(self, model: str) -> str:
        """
        Extract chatbot name from model string.

        Expected format: chatbot-{name} or {provider}/chatbot-{name}

        Args:
            model: Model name from the request

        Returns:
            Chatbot configuration name

        Raises:
            ValueError: If model format is invalid or chatbot not found
        """
        # Handle both "chatbot-edison" and "chatbot-llm/chatbot-edison" formats
        if "/" in model:
            model = model.split("/")[-1]

        # Extract chatbot name from "chatbot-{name}" format
        match = re.match(r"chatbot-(.+)", model)
        if not match:
            raise ValueError(
                f"Invalid model format: {model}. Expected: chatbot-{{name}}"
            )

        chatbot_name = match.group(1)

        # Map some common variations
        name_mappings = {
            "edison-guardrails": "edison_guardrails",
            "edison_guardrails": "edison_guardrails",
            "edison": "edison",
        }

        chatbot_name = name_mappings.get(chatbot_name, chatbot_name)

        if chatbot_name not in self.chatbot_configs:
            available = list(self.chatbot_configs.keys())
            raise ValueError(
                f"Unknown chatbot '{chatbot_name}'. Available: {available}"
            )

        return chatbot_name

    def _get_chatbot_config(self, model: str) -> Dict[str, Any]:
        """Get chatbot configuration based on model name."""
        chatbot_name = self._extract_chatbot_name(model)
        return self.chatbot_configs[chatbot_name]

    def completion(self, *args, **kwargs) -> Union[ModelResponse, Any]:
        """Synchronous completion method."""
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        user_message = self._extract_user_message(messages)

        try:
            config = self._get_chatbot_config(model)

            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{config['base_url']}{config['endpoint']}",
                    params={config["message_param"]: user_message},
                )
                response.raise_for_status()
                api_response = response.json()

            # Extract response text - handle different response formats
            response_text = self._extract_response_text(
                api_response, config["api_name"]
            )

            return litellm.completion(
                model="gpt-3.5-turbo",  # Use a standard model for the mock response
                messages=[{"role": "user", "content": user_message}],
                mock_response=response_text,
            )
        except Exception as e:
            logger.error(f"Chatbot API Error (model: {model}): {str(e)}")
            error_msg = f"Chatbot API Error: {str(e)}"
            return litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_message}],
                mock_response=error_msg,
            )

    async def acompletion(self, *args, **kwargs) -> Union[ModelResponse, Any]:
        """Asynchronous completion method."""
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        user_message = self._extract_user_message(messages)

        try:
            config = self._get_chatbot_config(model)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{config['base_url']}{config['endpoint']}",
                    params={config["message_param"]: user_message},
                )
                response.raise_for_status()
                api_response = response.json()

            # Extract response text - handle different response formats
            response_text = self._extract_response_text(
                api_response, config["api_name"]
            )

            return litellm.completion(
                model="gpt-3.5-turbo",  # Use a standard model for the mock response
                messages=[{"role": "user", "content": user_message}],
                mock_response=response_text,
            )
        except Exception as e:
            logger.error(f"Chatbot API Error (model: {model}): {str(e)}")
            error_msg = f"Chatbot API Error: {str(e)}"
            return litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_message}],
                mock_response=error_msg,
            )

    def _extract_user_message(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the last user message from the messages array."""
        user_message = ""

        # Look for the last user message (in case of conversation history)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Handle both string and list formats
                if isinstance(content, str):
                    user_message = content
                elif isinstance(content, list):
                    # For Anthropic-style content with text blocks
                    text_parts = [
                        part.get("text", "")
                        for part in content
                        if part.get("type") == "text"
                    ]
                    user_message = " ".join(text_parts)
                break

        return user_message or "Hello"

    def _extract_response_text(
        self, api_response: Dict[str, Any], api_name: str
    ) -> str:
        """
        Extract response text from API response, handling different formats.

        Common formats:
        - {"response": "text"}
        - {"message": "text"}
        - {"content": "text"}
        - {"text": "text"}
        """
        # Try common response field names
        for field in ["response", "message", "content", "text", "answer"]:
            if field in api_response:
                return str(api_response[field])

        # If no common field found, try to stringify the whole response
        logger.warning(f"Unknown response format from {api_name}: {api_response}")
        return str(api_response)


# Create the unified handler instance
chatbot_llm = UnifiedChatbotLLM()


def list_supported_models() -> List[str]:
    """
    List all supported chatbot models.

    Returns:
        List of model names in the format: chatbot-{name}
    """
    from .config import CHATBOT_CONFIGS

    return [f"chatbot-{name}" for name in CHATBOT_CONFIGS.keys()]


def validate_model_name(model: str) -> bool:
    """
    Validate if a model name is supported.

    Args:
        model: Model name to validate

    Returns:
        True if model is supported, False otherwise
    """
    try:
        chatbot_llm._extract_chatbot_name(model)
        return True
    except ValueError:
        return False


class RAGChatbotLLM(CustomLLM):
    """
    Custom LiteLLM handler for RAG chatbot that preserves all response fields.

    This handler uses the OpenAI client to call the RAG server and returns
    the raw response without filtering any custom fields like 'context' with citations.
    """

    def __init__(self, timeout: float = 120.0):
        """
        Initialize the RAG chatbot LLM handler.

        Args:
            timeout: Request timeout in seconds (default: 120.0)
        """
        super().__init__()
        self.timeout = timeout

    def completion(self, *args, **kwargs) -> Union[ModelResponse, Any]:
        """Synchronous completion method."""
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        api_base = kwargs.get("api_base")
        api_key = kwargs.get("api_key")
        temperature = kwargs.get("temperature")
        extra_body = kwargs.get("extra_body", {})

        if not api_base:
            raise ValueError("api_base is required for RAG chatbot")

        try:
            # Initialize OpenAI client with RAG server endpoint
            client = OpenAI(base_url=api_base, api_key=api_key)

            # Make the request using OpenAI client
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                extra_body=extra_body if extra_body else None,
            )

            # Return raw response dict with all fields preserved
            return (
                response.model_dump()
                if hasattr(response, "model_dump")
                else response.__dict__
            )

        except Exception as e:
            logger.error(f"RAG Chatbot API Error: {str(e)}")
            raise

    async def acompletion(self, *args, **kwargs) -> Union[ModelResponse, Any]:
        """Asynchronous completion method."""
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        api_base = kwargs.get("api_base")
        api_key = kwargs.get("api_key")
        temperature = kwargs.get("temperature")
        extra_body = kwargs.get("extra_body", {})

        if not api_base:
            raise ValueError("api_base is required for RAG chatbot")

        try:
            # Initialize OpenAI client with RAG server endpoint
            client = OpenAI(base_url=api_base, api_key=api_key)

            # Make the request using OpenAI client
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                extra_body=extra_body if extra_body else None,
            )

            # Return raw response dict with all fields preserved
            return (
                response.model_dump()
                if hasattr(response, "model_dump")
                else response.__dict__
            )

        except Exception as e:
            logger.error(f"RAG Chatbot API Error: {str(e)}")
            raise


# Create handler instances
chatbot_llm = UnifiedChatbotLLM()
rag_chatbot_llm = RAGChatbotLLM()
