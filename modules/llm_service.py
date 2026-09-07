import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from loguru import logger
from typing import AsyncGenerator
from pipecat.services.llm_service import LLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from modules.latency_tracker import tracker

class Gemma4E2BService(LLMService):
    """
    Pipecat LLM Service integrating Gemma 4 E2B Q4 GGUF via llama-cpp-python.
    Utilizes GPU layers on Kaggle Tesla T4 for ultra-low TTFT and streaming token delivery.
    """
    def __init__(
        self,
        repo_id: str = "google/gemma-4-E2B-it-qat-q4_0-gguf",
        filename: str = "gemma-4-E2B_q4_0-it.gguf",
        n_ctx: int = 2048,
        n_gpu_layers: int = 99,
        temperature: float = 0.6,
        max_tokens: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._repo_id = repo_id
        self._filename = filename
        self._temperature = temperature
        self._max_tokens = max_tokens

        logger.info(f"Downloading/verifying Gemma 4 E2B Q4 GGUF from {self._repo_id}...")
        model_path = hf_hub_download(repo_id=self._repo_id, filename=self._filename)
        logger.info(f"Loading Gemma 4 E2B into LlamaCpp from {model_path} (gpu_layers={n_gpu_layers})...")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        logger.info("Gemma 4 E2B Q4 loaded successfully.")

    async def _process_context(self, context: LLMContext) -> AsyncGenerator[Frame | None, None]:
        """
        Formats conversation history and streams tokens directly to the pipeline.
        """
        tracker.on_llm_start()
        yield LLMFullResponseStartFrame()

        messages = context.get_messages()

        try:
            # llama-cpp-python create_chat_completion streaming
            response_stream = self.llm.create_chat_completion(
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True,
            )

            for chunk in response_stream:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    token_text = delta.get("content", "")
                    if token_text:
                        tracker.on_llm_token()
                        yield TextFrame(text=token_text)

        except Exception as e:
            logger.error(f"Error during Gemma 4 E2B inference: {e}")
        finally:
            tracker.on_llm_end()
            yield LLMFullResponseEndFrame()
