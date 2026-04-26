import logging
import time
import asyncio
import os
from typing import Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from ..config import settings
from  dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

try:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
except Exception:  # pragma: no cover - import guard for optional dependency
    HuggingFaceEndpoint = None
    ChatHuggingFace = None

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - import guard for optional dependency
    ChatOpenAI = None


def resolve_hf_token() -> Optional[str]:
    """Resolve HF token with the same fallback aliases used in test scripts."""
    return (
        os.getenv("HUGGINGFACEHUB_ACCESS_KEY") 
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or settings.huggingface_api_token
        or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    )


def extract_content(response: Any) -> str:
    """Extract content from LLM response, handling both string and object responses."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


class LLMFactory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMFactory, cls).__new__(cls)
        return cls._instance

    def get_llm(self, temperature: float = 0.7, tier: str = "flash") -> Optional[BaseChatModel]:
        """
        Get an LLM instance based on performance tier (heavy or flash).
        Priority: HuggingFace (Primary) -> OpenRouter (Fallback) -> OpenAI (Final Fallback)
        """
        models = []
        hf_token = resolve_hf_token()
        
        # Determine the target models based on tier
        if tier == "heavy":
            hf_model = settings.hf_heavy_model
            or_model = settings.or_heavy_model
            logger.info(f"Requesting HEAVY tier - HF: {hf_model}, OR: {or_model}")
        else:
            hf_model = settings.hf_flash_model
            or_model = settings.or_flash_model
            logger.info(f"Requesting FLASH tier - HF: {hf_model}, OR: {or_model}")

        # 1. Try HuggingFace (Primary)
        if hf_token and HuggingFaceEndpoint and ChatHuggingFace:
            try:
                hf_endpoint = HuggingFaceEndpoint(
                    model=hf_model,
                    huggingfacehub_api_token=hf_token,
                    temperature=temperature,
                    verbose=True
                )
                # Wrap endpoint as a chat model so message-based async calls work correctly.
                hf_llm = ChatHuggingFace(llm=hf_endpoint)
                models.append(hf_llm)
                logger.info(f"Initialized HuggingFace model: {hf_model}")
            except Exception as e:
                logger.warning(f"HuggingFace model failed for {hf_model}: {str(e)}")

            # Flash-tier models may be non-chat for some providers; keep a chat-capable
            # HuggingFace heavy model as an internal fallback before external providers.
            if tier != "heavy" and settings.hf_heavy_model != hf_model:
                try:
                    hf_heavy_endpoint = HuggingFaceEndpoint(
                        model=settings.hf_heavy_model,
                        huggingfacehub_api_token=hf_token,
                        temperature=temperature,
                        verbose=True,
                    )
                    hf_heavy_llm = ChatHuggingFace(llm=hf_heavy_endpoint)
                    models.append(hf_heavy_llm)
                    logger.info(f"Initialized HuggingFace fallback model: {settings.hf_heavy_model}")
                except Exception as e:
                    logger.warning(
                        f"HuggingFace fallback model failed for {settings.hf_heavy_model}: {str(e)}"
                    )
        elif hf_token and (not HuggingFaceEndpoint or not ChatHuggingFace):
            logger.warning("HuggingFace token found but langchain_huggingface is not installed.")

        # 2. Try OpenRouter (Fallback)
        if settings.openrouter_api_key and ChatOpenAI:
            try:
                openrouter_llm = ChatOpenAI(
                    model=or_model,
                    api_key=settings.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=temperature,
                    default_headers={
                        "HTTP-Referer": "https://github.com/YTA-System",
                        "X-Title": "YTA-System"
                    }
                )
                models.append(openrouter_llm)
                logger.info(f"Initialized OpenRouter model: {or_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter model {or_model}: {str(e)}")
        elif settings.openrouter_api_key and not ChatOpenAI:
            logger.warning("OpenRouter key found but langchain_openai is not installed.")

        # 3. Add OpenRouter fallback model (if different from primary)
        if settings.openrouter_api_key and ChatOpenAI and settings.or_fallback_model != or_model:
            try:
                fallback_llm = ChatOpenAI(
                    model=settings.or_fallback_model,
                    api_key=settings.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=temperature
                )
                models.append(fallback_llm)
                logger.info(f"Initialized OpenRouter fallback: {settings.or_fallback_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter fallback: {str(e)}")

        # 4. Final Fallback to standard OpenAI if available
        if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here" and ChatOpenAI:
            try:
                openai_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=settings.openai_api_key, temperature=temperature)
                models.append(openai_llm)
                logger.info("Initialized OpenAI fallback: gpt-3.5-turbo")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI fallback: {str(e)}")
        elif settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here" and not ChatOpenAI:
            logger.warning("OpenAI key found but langchain_openai is not installed.")

        if not models:
            logger.error("No valid LLM configuration found!")
            return None
            
        primary_llm = models[0]
        if len(models) > 1:
            primary_llm = primary_llm.with_fallbacks(models[1:])
            logger.info(f"Primary LLM: {type(primary_llm).__name__} with {len(models)-1} fallbacks")
            
        return primary_llm

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

    async def ainvoke_with_retry(self, llm: BaseChatModel, messages, max_retries: int = 3, base_delay: float = 5.0):
        """Invoke LLM asynchronously with exponential backoff retry for rate limits"""
        for attempt in range(max_retries):
            try:
                return await llm.ainvoke(messages)
            except Exception as e:
                error_str = str(e)
                is_async_stop_iteration = "coroutine raised StopIteration" in error_str
                if is_async_stop_iteration:
                    logger.warning("Async LLM call raised StopIteration; retrying via sync invoke in a worker thread.")
                    return await asyncio.to_thread(llm.invoke, messages)
                is_rate_limit = "429" in error_str or "rate" in error_str.lower() or "Too Many" in error_str
                if is_rate_limit and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited (async attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise

llm_factory = LLMFactory()
