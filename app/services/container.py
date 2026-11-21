"""Models loaded once at startup and shared by every session."""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.services.llm import OllamaLLM
from app.services.stt import _Base as STT
from app.services.stt import create_stt
from app.services.tts import _Base as TTS
from app.services.tts import create_tts
from app.services.vad import load_vad_model

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    stt: STT
    llm: OllamaLLM
    tts: TTS
    # One model on 16 GB: serve one reply at a time, others wait in line.
    llm_lock: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))

    def new_vad_model(self):
        return load_vad_model()

    @classmethod
    async def create(cls, s: Settings) -> "Services":
        t = time.perf_counter()
        loop = asyncio.get_running_loop()
        stt = await loop.run_in_executor(
            None, create_stt, s.stt_engine, s.stt_model, s.stt_language, s.stt_beam_size, s.stt_prompt, s.stt_compute_type
        )
        log.info("STT ready (%s %s) in %.1fs", stt.name, s.stt_model, time.perf_counter() - t)

        t = time.perf_counter()
        tts = await loop.run_in_executor(None, create_tts, s.tts_engine, s.tts_voice, s.say_voice, s.tts_device)
        log.info("TTS ready (%s) in %.1fs", tts.name, time.perf_counter() - t)

        t = time.perf_counter()
        llm = OllamaLLM(s.ollama_url, s.llm_model, s.llm_max_tokens, s.llm_keep_alive, s.llm_num_ctx)
        try:
            await llm.warmup()
            log.info("LLM ready (%s) in %.1fs", s.llm_model, time.perf_counter() - t)
        except Exception as e:
            log.error("LLM warmup failed: %s. Is Ollama running? (brew services start ollama)", e)

        load_vad_model()  # fetch/validate VAD weights before first connection
        return cls(settings=s, stt=stt, llm=llm, tts=tts)

    async def close(self):
        self.stt.close()
        self.tts.close()
        await self.llm.close()
