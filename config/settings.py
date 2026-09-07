import os
from pydantic import BaseModel, Field

class PipelineSettings(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "7860")))
    ice_servers: list[str] = Field(
        default_factory=lambda: [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
        ]
    )

    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_frame_duration_ms: int = 20

    vad_confidence: float = 0.6
    vad_stop_secs: float = 0.5
    vad_start_secs: float = 0.15
    enable_rnnoise: bool = True

    # ASR settings (Qwen3-ASR)
    asr_model_id: str = Field(
        default_factory=lambda: os.getenv("ASR_MODEL_ID", "Qwen/Qwen3-ASR-0.6B-hf")
    )
    asr_device: str = Field(default_factory=lambda: os.getenv("ASR_DEVICE", "cuda"))
    asr_language: str = "Hindi"

    # LLM settings (Gemma 4 E2B Q4 GGUF)
    llm_repo_id: str = Field(
        default_factory=lambda: os.getenv("LLM_REPO_ID", "google/gemma-4-E2B-it-qat-q4_0-gguf")
    )
    llm_filename: str = Field(
        default_factory=lambda: os.getenv("LLM_FILENAME", "gemma-4-E2B_q4_0-it.gguf")
    )
    llm_n_ctx: int = 2048
    llm_n_gpu_layers: int = 99
    llm_temperature: float = 0.6
    llm_max_tokens: int = 128

    system_prompt: str = Field(
        default_factory=lambda: os.getenv(
            "SYSTEM_PROMPT",
            "Aap ek professional, polite aur helpful Hindi customer service assistant hain. "
            "Aapko bilkul short, concise aur natural spoken Hindi mein jawab dena hai. "
            "Jawab hamesha 1-2 chote sentences mein dein taaki conversation natural lage. "
            "Emojis ya markdown formatting ka use bilkul na karein."
        )
    )

    # TTS settings (MMS-TTS-Hindi)
    tts_model_id: str = Field(
        default_factory=lambda: os.getenv("TTS_MODEL_ID", "facebook/mms-tts-hin")
    )
    tts_device: str = Field(default_factory=lambda: os.getenv("TTS_DEVICE", "cuda"))
    tts_sample_rate: int = 16000

settings = PipelineSettings()
