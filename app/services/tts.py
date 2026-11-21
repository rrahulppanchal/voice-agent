"""Text-to-speech engines. Each returns int16 mono PCM at `sample_rate`."""

import asyncio
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


class _Base:
    sample_rate = 24000
    name = "base"

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    def _synth(self, text: str) -> np.ndarray:
        raise NotImplementedError

    async def synth(self, text: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.pool, self._synth, text)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def _use_system_espeak():
    """The espeak-ng bundled with kokoro's dependency has a broken data path on macOS.
    Point it at a Homebrew install (`brew install espeak-ng`) when one exists."""
    import espeakng_loader

    for prefix in ("/opt/homebrew", "/usr/local"):
        lib = f"{prefix}/lib/libespeak-ng.dylib"
        data = f"{prefix}/share/espeak-ng-data"
        if os.path.exists(lib) and os.path.isdir(data):
            espeakng_loader.get_library_path = lambda lib=lib: lib
            espeakng_loader.get_data_path = lambda data=data: data
            return


class KokoroTTS(_Base):
    name = "kokoro"

    def __init__(self, voice: str, device: str = "cpu"):
        super().__init__()
        _use_system_espeak()
        if device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        from kokoro import KModel, KPipeline

        self.voice = voice
        model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
        self.pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)

    def _synth(self, text: str) -> np.ndarray:
        parts = [r.audio.numpy() for r in self.pipe(text, voice=self.voice) if r.audio is not None]
        if not parts:
            return np.zeros(0, dtype=np.int16)
        audio = np.concatenate(parts)
        return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


class SayTTS(_Base):
    """macOS built-in voice. Zero downloads; used as a fallback."""

    name = "say"

    def __init__(self, voice: str = ""):
        super().__init__()
        self.voice = voice

    def _synth(self, text: str) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            cmd = ["say", "-o", f.name, "--file-format=WAVE", f"--data-format=LEI16@{self.sample_rate}"]
            if self.voice:
                cmd += ["-v", self.voice]
            subprocess.run(cmd + ["--", text], check=True, capture_output=True)
            audio, _ = sf.read(f.name, dtype="int16")
        return audio if audio.ndim == 1 else audio[:, 0]


def create_tts(engine: str, kokoro_voice: str, say_voice: str, device: str = "cpu") -> _Base:
    if engine in ("auto", "kokoro"):
        try:
            tts = KokoroTTS(kokoro_voice, device)
            # fail fast if phonemizer or weights are broken; also warms up short and long inputs
            for text in ("Ready.", "This is a short warm up.", "Warming up the voice with a longer sentence now."):
                tts._synth(text)
            return tts
        except Exception as e:
            if engine == "kokoro":
                raise
            log.warning("Kokoro unavailable (%s); falling back to macOS say", e)
    return SayTTS(say_voice)
