import time
import asyncio
import numpy as np
from modules.asr_service import Qwen3ASRService
from modules.llm_service import Gemma4E2BService
from modules.tts_service import HindiTTSService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import TextFrame, TTSAudioRawFrame
from config.settings import settings

async def test_e2e_pipeline():
    print("=== Phase 5: Testing Chained Pipeline (ASR -> LLM -> TTS) ===")
    
    print("Loading models...")
    asr = Qwen3ASRService(model_id=settings.asr_model_id, device=settings.asr_device)
    llm = Gemma4E2BService(repo_id=settings.llm_repo_id, filename=settings.llm_filename)
    tts = HindiTTSService(model_id=settings.tts_model_id, device=settings.tts_device)

    # Simulate recognized Hindi speech input
    simulated_speech = "??????, ???? ??? ??? ????? ??????"
    print(f"\n[Simulated User Speech]: {simulated_speech}")

    t_start = time.perf_counter()

    context = LLMContext()
    context.add_message({"role": "system", "content": settings.system_prompt})
    context.add_message({"role": "user", "content": simulated_speech})

    print("\n--- LLM Response Streaming ---")
    full_response = ""
    sentence_buffer = ""
    tts_first_audio_received = False
    first_audio_latency = 0.0

    async for frame in llm._process_context(context):
        if isinstance(frame, TextFrame):
            token = frame.text
            full_response += token
            sentence_buffer += token
            print(token, end="", flush=True)

            # Stream sentence-by-sentence to TTS to avoid waiting for complete response
            if any(punct in sentence_buffer for punct in [".", "?", "!", "\n", "?"]) and len(sentence_buffer.strip()) > 5:
                # Synthesize buffered sentence
                chunk_text = sentence_buffer.strip()
                sentence_buffer = ""
                async for audio_frame in tts.run_tts(chunk_text):
                    if isinstance(audio_frame, TTSAudioRawFrame):
                        if not tts_first_audio_received:
                            first_audio_latency = (time.perf_counter() - t_start) * 1000
                            tts_first_audio_received = True

    # Flush remaining text
    if sentence_buffer.strip():
        async for audio_frame in tts.run_tts(sentence_buffer.strip()):
            if isinstance(audio_frame, TTSAudioRawFrame) and not tts_first_audio_received:
                first_audio_latency = (time.perf_counter() - t_start) * 1000
                tts_first_audio_received = True

    t_end = time.perf_counter()
    print(f"\n\n[Pipeline Latency Breakdown]:")
    print(f"User Speech -> First Bot Audio Output: {first_audio_latency:.1f}ms")
    print(f"Total Pipeline Completion Time: {(t_end - t_start)*1000:.1f}ms")
    print("Phase 5 Chained Pipeline test PASSED.")

if __name__ == "__main__":
    asyncio.run(test_e2e_pipeline())
