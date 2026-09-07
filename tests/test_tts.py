import time
import asyncio
from modules.tts_service import HindiTTSService
from pipecat.frames.frames import TTSAudioRawFrame
from config.settings import settings

async def test_tts_standalone():
    print("=== Phase 4: Testing Hindi TTS (MMS-TTS-Hindi) Standalone ===")
    tts = HindiTTSService(
        model_id=settings.tts_model_id,
        device=settings.tts_device,
        sample_rate=16000
    )

    test_hindi_text = "??????, ???? ???? ??? ??????? ???? ???? ???? ???? ?? ??? ???"
    print(f"Synthesizing Hindi text: '{test_hindi_text}'...")
    
    t0 = time.perf_counter()
    chunks = 0
    total_bytes = 0
    first_chunk_time = 0

    async for frame in tts.run_tts(test_hindi_text):
        if isinstance(frame, TTSAudioRawFrame):
            if chunks == 0:
                first_chunk_time = time.perf_counter()
            chunks += 1
            total_bytes += len(frame.audio)

    t1 = time.perf_counter()
    print(f"Time to first audio: {(first_chunk_time - t0)*1000:.2f}ms")
    print(f"Total synthesis time: {(t1 - t0)*1000:.2f}ms for {chunks} 20ms chunks ({total_bytes} bytes PCM)")
    print("TTS Standalone test PASSED.")

if __name__ == "__main__":
    asyncio.run(test_tts_standalone())
