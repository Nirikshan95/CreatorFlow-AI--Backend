import logging
import time
from typing import Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from ..config import settings

logger = logging.getLogger(__name__)

class LLMFactory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMFactory, cls).__new__(cls)
        return cls._instance

    def get_llm(self, temperature: float = 0.7) -> Optional[BaseChatModel]:
        """
        Get an LLM instance, prioritizing HuggingFace Hub and falling back to OpenRouter.
        """
        # 1. Try HuggingFace Hub
        if settings.huggingface_api_token:
            try:
                logger.info(f"Initializing LLM via HuggingFace Hub: {settings.primary_model}")
                llm = HuggingFaceEndpoint(
                    repo_id=settings.primary_model,
                    huggingfacehub_api_token=settings.huggingface_api_token,
                    temperature=temperature,
                    max_new_tokens=2048,
                )
                chat_llm = ChatHuggingFace(llm=llm)
                logger.info("Successfully initialized HuggingFace via ChatHuggingFace")
                return chat_llm
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFace Hub: {str(e)}. Falling back to OpenRouter...")

        # 2. Try OpenRouter
        if settings.openrouter_api_key:
            try:
                logger.info(f"Initializing LLM via OpenRouter: {settings.fallback_model}")
                return ChatOpenAI(
                    model=settings.fallback_model,
                    api_key=settings.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=temperature,
                    default_headers={
                        "HTTP-Referer": "https://github.com/YTA-System",
                        "X-Title": "YTA-System"
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter: {str(e)}")

        # 3. Final Fallback to standard OpenAI if available
        if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here":
            logger.info("Falling back to direct OpenAI API")
            return ChatOpenAI(model="gpt-3.5-turbo", api_key=settings.openai_api_key)

        logger.error("No valid LLM configuration found! Agents will run in mock mode.")
        return None

    def invoke_with_retry(self, llm: BaseChatModel, messages, max_retries: int = 3, base_delay: float = 5.0):
        """Invoke LLM with exponential backoff retry for rate limits"""
        for attempt in range(max_retries):
            try:
                return llm.invoke(messages)
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate" in error_str.lower() or "Too Many" in error_str
                if is_rate_limit and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise

llm_factory = LLMFactory()
