import io
import wave
import torch
import numpy as np
from typing import AsyncGenerator
from loguru import logger

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
    Pipecat STT Service supporting Qwen3-ASR via `qwen_asr` or Hugging Face transformers,
    with an ultra-fast fallback to Hindi-tuned Whisper if transformers lacks qwen3_asr.
    """
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-0.6B",
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
        self.backend = None

        logger.info(f"Initializing Hindi ASR: {self._model_id} on {self._device}...")

        # 1. Try official qwen_asr library
        try:
            from qwen_asr import Qwen3ASRModel
            logger.info("Loading Qwen3-ASR via qwen_asr.Qwen3ASRModel...")
            self.model = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-0.6B",
                device_map="auto" if self._device == "cuda" else None,
                dtype=torch.float16 if self._device == "cuda" else torch.float32,
                max_new_tokens=128
            )
            self.backend = "qwen_asr"
            logger.info("Qwen3-ASR loaded successfully via qwen_asr.")
            return
        except Exception as e:
            logger.warning(f"Could not load via qwen_asr: {e}. Trying transformers...")

        # 2. Try Hugging Face AutoModelForMultimodalLM
        try:
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            self.processor = AutoProcessor.from_pretrained(self._model_id)
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
                device_map="auto" if self._device == "cuda" else None
            )
            self.model.eval()
            self.backend = "transformers_qwen"
            logger.info("Qwen3-ASR loaded successfully via transformers.")
            return
        except Exception as e:
            logger.warning(f"Could not load Qwen3-ASR in transformers ({e}). Loading fast Hindi Whisper fallback...")

        # 3. Fallback to OpenAI Whisper (Hindi configured)
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            fallback_id = "openai/whisper-small"
            self.processor = WhisperProcessor.from_pretrained(fallback_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(fallback_id).to(self._device)
            self.model.eval()
            self.backend = "transformers_whisper"
            logger.info(f"Loaded {fallback_id} for Hindi ASR successfully.")
        except Exception as e:
            logger.error(f"Failed to load Hindi ASR fallback: {e}")
            raise e

    def transcribe_audio_np(self, audio_data: np.ndarray) -> str:
        """
        Transcribes a 1D numpy float32 array sampled at self._sample_rate.
        """
        if len(audio_data) < self._sample_rate * 0.2:  # Less than 200ms
            return ""

        tracker.on_asr_start()
        try:
            if self.backend == "qwen_asr":
                results = self.model.transcribe(audio=audio_data, language="Hindi")
                if isinstance(results, list) and len(results) > 0:
                    text = results[0].get("text", "")
                elif isinstance(results, dict):
                    text = results.get("text", "")
                else:
                    text = str(results)
            elif self.backend == "transformers_whisper":
                input_features = self.processor(
                    audio_data, sampling_rate=self._sample_rate, return_tensors="pt"
                ).input_features.to(self.model.device)
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="hi", task="transcribe")
                with torch.no_grad():
                    predicted_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
                text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            else:
                # transformers_qwen
                inputs = self.processor.apply_transcription_request(
                    audio=audio_data,
                    sampling_rate=self._sample_rate
                )
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.model.device)
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=128)
                generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                try:
                    text = self.processor.decode(generated_ids, return_format="transcription_only")[0].strip()
                except Exception:
                    text = self.processor.decode(generated_ids, skip_special_tokens=True)[0].strip()

            tracker.on_asr_end(text)
            return text
        except Exception as e:
            logger.error(f"Error during ASR inference: {e}")
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
