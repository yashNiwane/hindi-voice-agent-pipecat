import time
from loguru import logger

class LatencyTracker:
    """
    Tracks and instruments latency across all pipeline stages:
    1. VAD / User Speech End -> ASR complete
    2. ASR complete -> LLM Time to First Token (TTFT)
    3. LLM complete -> TTS Time to First Audio (TTFA)
    4. Total Conversational Latency (User stopped speaking -> First bot audio played)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.user_speech_start_time: float = 0.0
        self.user_speech_stop_time: float = 0.0
        self.asr_start_time: float = 0.0
        self.asr_end_time: float = 0.0
        self.llm_start_time: float = 0.0
        self.llm_first_token_time: float = 0.0
        self.llm_end_time: float = 0.0
        self.tts_start_time: float = 0.0
        self.tts_first_audio_time: float = 0.0
        self.tts_end_time: float = 0.0
        self.tokens_generated: int = 0

    def on_user_speech_stopped(self):
        self.reset()
        self.user_speech_stop_time = time.perf_counter()

    def on_asr_start(self):
        self.asr_start_time = time.perf_counter()

    def on_asr_end(self, text: str):
        self.asr_end_time = time.perf_counter()
        dur = (self.asr_end_time - self.asr_start_time) * 1000
        logger.info(f"[LATENCY] ASR Processing: {dur:.1f}ms | Recognized: '{text}'")

    def on_llm_start(self):
        self.llm_start_time = time.perf_counter()

    def on_llm_token(self):
        if self.tokens_generated == 0:
            self.llm_first_token_time = time.perf_counter()
            ttft = (self.llm_first_token_time - self.llm_start_time) * 1000
            logger.info(f"[LATENCY] LLM TTFT (Time-To-First-Token): {ttft:.1f}ms")
        self.tokens_generated += 1

    def on_llm_end(self):
        self.llm_end_time = time.perf_counter()
        dur = (self.llm_end_time - self.llm_start_time) * 1000
        tps = (self.tokens_generated / (dur / 1000)) if dur > 0 else 0
        logger.info(f"[LATENCY] LLM Total Generation: {dur:.1f}ms | Tokens: {self.tokens_generated} ({tps:.1f} tok/s)")

    def on_tts_start(self):
        self.tts_start_time = time.perf_counter()

    def on_tts_first_audio(self):
        if self.tts_first_audio_time == 0.0:
            self.tts_first_audio_time = time.perf_counter()
            ttfa = (self.tts_first_audio_time - self.tts_start_time) * 1000
            logger.info(f"[LATENCY] TTS TTFA (Time-To-First-Audio): {ttfa:.1f}ms")
            
            if self.user_speech_stop_time > 0:
                e2e = (self.tts_first_audio_time - self.user_speech_stop_time) * 1000
                logger.info(f"[LATENCY] >>> TOTAL END-TO-END CONVERSATIONAL LATENCY: {e2e:.1f}ms <<<")

    def on_tts_end(self):
        self.tts_end_time = time.perf_counter()
        dur = (self.tts_end_time - self.tts_start_time) * 1000
        logger.info(f"[LATENCY] TTS Total Synthesis: {dur:.1f}ms")

tracker = LatencyTracker()
