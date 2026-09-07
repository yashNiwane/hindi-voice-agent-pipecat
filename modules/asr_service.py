import io
import wave
import torch
import numpy as np
from typing import AsyncGenerator
from loguru import logger
from transformers import AutoProcessor, AutoModelForMultimodalLM

from pipecat.services.stt_service import STTService
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from modules.latency_tracker import tracker

class Qwen3ASRService(STTService):
    """
    Pipecat STT Service integrating Qwen3-ASR (0.6B / 1.7B)
    Runs offline on Kaggle GPU with Hugging Face transformers.
    Transcribes audio into Hindi with high accuracy and low latency.
    """
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-0.6B-hf",
        device: str = "cuda",
        language: str = "Hindi",
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._model_id = model_id
        self._device = device if torch.cuda.is_available() else "cpu"
        self._language = language
        self._sample_rate = sample_rate

        logger.info(f"Loading Qwen3-ASR model: {self._model_id} on {self._device}...")
        self.processor = AutoProcessor.from_pretrained(self._model_id)
        
        dtype = torch.bfloat16 if (self._device != "cpu" and torch.cuda.is_bf16_supported()) else (torch.float16 if self._device != "cpu" else torch.float32)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None
        )
        if self._device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        logger.info("Qwen3-ASR loaded successfully.")

    def transcribe_audio_np(self, audio_data: np.ndarray) -> str:
        """
        Transcribes a 1D numpy float32 array sampled at self._sample_rate.
        """
        if len(audio_data) < self._sample_rate * 0.2:  # Less than 200ms
            return ""

        tracker.on_asr_start()
        try:
            # Prepare inputs using the official Qwen3-ASR AutoProcessor
            inputs = self.processor.apply_transcription_request(
                audio=audio_data,
                sampling_rate=self._sample_rate
            )
            # Move to model device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128
                )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            
            # Decode transcription
            try:
                text = self.processor.decode(generated_ids, return_format="transcription_only")[0].strip()
            except Exception:
                text = self.processor.decode(generated_ids, skip_special_tokens=True)[0].strip()

            tracker.on_asr_end(text)
            return text
        except Exception as e:
            logger.error(f"Error during Qwen3-ASR inference: {e}")
            return ""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """
        Receives accumulated audio buffer from VAD upon user speech stop.
        """
        if not audio or len(audio) == 0:
            return

        # Audio is 16-bit PCM mono @ self._sample_rate
        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        text = self.transcribe_audio_np(audio_np)
        
        if text:
            logger.info(f"[ASR Final]: {text}")
            yield TranscriptionFrame(text=text, user_id="user", timestamp=f"{tracker.asr_end_time}")
