from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

def get_vad_analyzer(confidence: float = 0.6, start_secs: float = 0.15, stop_secs: float = 0.5) -> SileroVADAnalyzer:
    """
    Configures Silero VAD for telecalling turn detection:
    - start_secs: 0.15s enables quick barge-in detection when user starts speaking.
    - stop_secs: 0.5s detects conversational turn completion without awkward prolonged silences.
    """
    logger.info(f"Initializing Silero VAD (confidence={confidence}, stop_secs={stop_secs}s, start_secs={start_secs}s)...")
    vad = SileroVADAnalyzer(
        params=VADParams(
            confidence=confidence,
            start_secs=start_secs,
            stop_secs=stop_secs,
        )
    )
    return vad

def get_noise_filter(enabled: bool = True):
    """
    Initializes RNNoise neural noise suppression filter.
    Separates background noise suppression from VAD and AGC.
    """
    if not enabled:
        return None
    try:
        from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
        logger.info("Initializing RNNoise neural noise filter...")
        return RNNoiseFilter()
    except Exception as e:
        logger.warning(f"RNNoise filter not loaded (fallback to standard audio): {e}")
        return None
