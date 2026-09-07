import time
import asyncio
from modules.llm_service import Gemma4E2BService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import TextFrame
from config.settings import settings

async def test_llm_standalone():
    print("=== Phase 3: Testing Gemma 4 E2B Q4 Standalone ===")
    llm = Gemma4E2BService(
        repo_id=settings.llm_repo_id,
        filename=settings.llm_filename,
        n_ctx=settings.llm_n_ctx,
        n_gpu_layers=settings.llm_n_gpu_layers,
        temperature=settings.llm_temperature,
        max_tokens=64
    )

    context = LLMContext()
    context.add_message({"role": "system", "content": settings.system_prompt})
    context.add_message({"role": "user", "content": "Namaste, mujhe apna internet plan recharge karwana hai." })

    print("Streaming response tokens from Gemma 4 E2B Q4:")
    t0 = time.perf_counter()
    tokens = []
    async for frame in llm._process_context(context):
        if isinstance(frame, TextFrame):
            tokens.append(frame.text)
            print(frame.text, end="", flush=True)
    t1 = time.perf_counter()
    print(f"\nCompleted generation in {(t1 - t0)*1000:.2f}ms. Total tokens: {len(tokens)}")
    print("LLM Standalone test PASSED.")

if __name__ == "__main__":
    asyncio.run(test_llm_standalone())
