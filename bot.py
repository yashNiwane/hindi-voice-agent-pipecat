import os
from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    LLMRunFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from config.settings import settings
from modules.asr_service import Qwen3ASRService
from modules.llm_service import Gemma4E2BService
from modules.tts_service import HindiTTSService
from modules.vad_noise import get_vad_analyzer, get_noise_filter
from modules.latency_tracker import tracker

load_dotenv(override=True)

# Global model instances for zero re-initialization latency across calls
_global_asr = None
_global_llm = None
_global_tts = None

def get_shared_models():
    global _global_asr, _global_llm, _global_tts
    if _global_asr is None:
        _global_asr = Qwen3ASRService(
            model_id=settings.asr_model_id,
            device=settings.asr_device,
            language=settings.asr_language,
            sample_rate=settings.audio_sample_rate
        )
    if _global_llm is None:
        _global_llm = Gemma4E2BService(
            repo_id=settings.llm_repo_id,
            filename=settings.llm_filename,
            n_ctx=settings.llm_n_ctx,
            n_gpu_layers=settings.llm_n_gpu_layers,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )
    if _global_tts is None:
        _global_tts = HindiTTSService(
            model_id=settings.tts_model_id,
            device=settings.tts_device,
            sample_rate=settings.tts_sample_rate
        )
    return _global_asr, _global_llm, _global_tts

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    asr, llm, tts = get_shared_models()

    vad_analyzer = get_vad_analyzer(
        confidence=settings.vad_confidence,
        start_secs=settings.vad_start_secs,
        stop_secs=settings.vad_stop_secs
    )
    noise_filter = get_noise_filter(enabled=settings.enable_rnnoise)

    # Initialize LLM Context and Conversation Memory
    context = LLMContext()
    context.add_message({
        "role": "system",
        "content": settings.system_prompt
    })

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer),
    )

    # Build the sequential streaming pipeline
    pipeline = Pipeline([
        transport.input(),
        asr,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"WebRTC client connected. Starting conversational session.")
        # Trigger Hindi greeting
        context.add_message({
            "role": "developer",
            "content": "Aap call pe connect ho chuke hain. Kripya namaste bolkar caller ka swagat karein aur poochhein ki aap unki kya sahayata kar sakte hain."
        })
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("WebRTC client disconnected.")
        await runner.cancel()

    await runner.run()

async def bot(runner_args: RunnerArguments):
    """Pipecat runner entry point."""
    webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
    noise_filter = get_noise_filter(enabled=settings.enable_rnnoise)

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=settings.audio_sample_rate,
            audio_out_sample_rate=settings.audio_sample_rate,
            audio_in_filter=noise_filter,
        ),
    )

    await run_bot(transport, runner_args)

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
