import torch
import numpy as np
from typing import AsyncGenerator
from loguru import logger
from transformers import VitsModel, AutoTokenizer

from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import (
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
)
from modules.latency_tracker import tracker

class HindiTTSService(TTSService):
    """
    Pipecat TTS Service using Facebook MMS-TTS-Hindi (VITS architecture).
    Synthesizes native, intelligible Hindi speech with low latency on Kaggle GPU.
    Outputs 16kHz 16-bit mono PCM audio frames chunked for WebRTC streaming playback.
    """
    def __init__(
        self,
        model_id: str = "facebook/mms-tts-hin",
        device: str = "cuda",
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._model_id = model_id
        self._device = device if torch.cuda.is_available() else "cpu"
        self._sample_rate = sample_rate

        logger.info(f"Loading Hindi TTS model: {self._model_id} on {self._device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self.model = VitsModel.from_pretrained(self._model_id)
        self.model = self.model.to(self._device)
        self.model.eval()
        logger.info("Hindi TTS model loaded successfully.")

    async def run_tts(self, text: str, context_id: str | None = None) -> AsyncGenerator[Frame | None, None]:
        """
        Synthesizes a chunk of text into streaming PCM audio frames.
        """
        text = text.strip()
        if not text:
            return

        tracker.on_tts_start()
        yield TTSStartedFrame(context_id=context_id)

        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self._device)

            with torch.no_grad():
                output = self.model(**inputs).waveform

            waveform = output.squeeze().cpu().numpy()
            
            # Normalize and convert float32 (-1.0 to 1.0) to 16-bit PCM bytes
            waveform_int16 = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
            pcm_bytes = waveform_int16.tobytes()

            # Stream audio in 20ms chunks (320 samples = 640 bytes @ 16kHz 16-bit mono)
            chunk_size = int(self._sample_rate * 0.02) * 2  # 640 bytes
            first_chunk = True

            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad the final chunk to align with 20ms WebRTC frame requirement
                    chunk = chunk + b'\x00' * (chunk_size - len(chunk))
                
                if first_chunk:
                    tracker.on_tts_first_audio()
                    first_chunk = False

                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self._sample_rate,
                    num_channels=1
                )

        except Exception as e:
            logger.error(f"Error in Hindi TTS synthesis: {e}")
        finally:
            tracker.on_tts_end()
            yield TTSStoppedFrame(context_id=context_id)
