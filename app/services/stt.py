"""Speech-to-text, run off the event loop.

Two engines:
- "mlx":            mlx-whisper on the Apple GPU. Fast enough to run large-v3-turbo (best accuracy,
                    handles accents and other languages). Ollama is idle while STT runs, so no contention.
- "faster-whisper": CTranslate2 on the CPU. Portable (Linux/Windows), but only small models are fast.
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Whisper sometimes "hears" these in noise; drop them when they are the whole utterance.
_HALLUCINATIONS = {"you", "thank you", "thanks for watching", "bye", "thank you for watching"}


def _clean(text: str) -> str:
    text = text.strip()
    bare = re.sub(r"[^\w\s]", "", text).strip().lower()
    return "" if not bare or bare in _HALLUCINATIONS else text


class _Base:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    def _transcribe(self, audio: np.ndarray) -> str:
        raise NotImplementedError

    async def transcribe(self, pcm: np.ndarray) -> str:
        audio = pcm.astype(np.float32) / 32768.0
        loop = asyncio.get_running_loop()
        return _clean(await loop.run_in_executor(self.pool, self._transcribe, audio))

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class FasterWhisperSTT(_Base):
    name = "faster-whisper"

    def __init__(self, model: str, language: str | None, beam_size: int, prompt: str, compute_type: str):
        super().__init__()
        from faster_whisper import WhisperModel

        self.language, self.beam_size, self.prompt = language, beam_size, prompt or None
        self.model = WhisperModel(model, device="cpu", compute_type=compute_type)

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            initial_prompt=self.prompt,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments)


class MlxWhisperSTT(_Base):
    name = "mlx"

    def __init__(self, model: str, language: str | None, prompt: str):
        super().__init__()
        import mlx_whisper

        self._mlx = mlx_whisper
        self.repo = model if "/" in model else f"mlx-community/whisper-{model}"
        self.language, self.prompt = language, prompt or None
        # load weights + compile kernels on the STT thread before the first request
        self.pool.submit(self._transcribe, np.zeros(16000, dtype=np.float32)).result()

    def _transcribe(self, audio: np.ndarray) -> str:
        out = self._mlx.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=self.language,
            initial_prompt=self.prompt,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        return out["text"]


def create_stt(engine: str, model: str, language: str, beam_size: int, prompt: str, compute_type: str) -> _Base:
    lang = language or None  # empty = auto-detect (needs a multilingual model, not *.en)
    if engine == "mlx":
        return MlxWhisperSTT(model, lang, prompt)
    return FasterWhisperSTT(model, lang, beam_size, prompt, compute_type)
