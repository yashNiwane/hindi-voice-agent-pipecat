import time
import torch
import numpy as np
from modules.asr_service import Qwen3ASRService
from config.settings import settings

def test_asr_standalone():
    print("=== Phase 2: Testing Qwen3-ASR Standalone ===")
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    asr = Qwen3ASRService(
        model_id=settings.asr_model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        language="Hindi",
        sample_rate=16000
    )
    
    # Generate 2 seconds of synthetic test sine tone
    duration = 2.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    synthetic_audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    print("Running ASR inference test...")
    t0 = time.perf_counter()
    transcription = asr.transcribe_audio_np(synthetic_audio)
    t1 = time.perf_counter()
    print(f"ASR completed in {(t1 - t0)*1000:.2f}ms. Output text: '{transcription}'")
    print("ASR Standalone test PASSED.")

if __name__ == "__main__":
    test_asr_standalone()
