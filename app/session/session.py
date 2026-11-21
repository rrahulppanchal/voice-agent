"""One VoiceSession per WebSocket connection.

Pipeline:  mic PCM -> VAD -> utterance -> STT -> LLM tokens -> sentences -> TTS -> audio out
State:     IDLE -> LISTENING -> THINKING -> SPEAKING -> IDLE   (user speech = barge-in, cancels reply)
"""

import asyncio
import contextlib
import json
import logging
import time
from enum import Enum

from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.services.chunker import SentenceChunker
from app.services.container import Services
from app.services.vad import VADSegmenter
from app.session.history import History

log = logging.getLogger(__name__)

AUDIO_CHUNK_MS = 100


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceSession:
    def __init__(self, ws: WebSocket, services: Services):
        s = services.settings
        self.ws = ws
        self.svc = services
        self.vad = VADSegmenter(
            services.new_vad_model(),
            threshold=s.vad_threshold,
            min_silence_ms=s.vad_min_silence_ms,
            speech_pad_ms=s.vad_speech_pad_ms,
            min_speech_ms=s.vad_min_speech_ms,
            max_speech_s=s.vad_max_speech_s,
        )
        self.history = History(s.system_prompt, s.max_history_turns)
        self.state = State.IDLE
        self.utterances: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=4)
        self.reply_task: asyncio.Task | None = None
        self.send_lock = asyncio.Lock()
        self.turn = 0
        self.closed = False

    # ---------- outbound ----------

    async def send_json(self, data: dict):
        if self.closed:
            return
        async with self.send_lock:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await self.ws.send_text(json.dumps(data))

    async def send_bytes(self, data: bytes):
        if self.closed:
            return
        async with self.send_lock:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await self.ws.send_bytes(data)

    async def set_state(self, state: State):
        if state != self.state:
            self.state = state
            await self.send_json({"type": "state", "state": state.value})

    # ---------- main loop ----------

    async def run(self):
        await self.send_json(
            {
                "type": "ready",
                "input_sample_rate": 16000,
                "output_sample_rate": self.svc.tts.sample_rate,
                "llm": self.svc.settings.llm_model,
                "tts": self.svc.tts.name,
            }
        )
        stt_task = asyncio.create_task(self._stt_loop())
        try:
            await self._recv_loop()
        finally:
            self.closed = True
            stt_task.cancel()
            await self._cancel_reply()
            with contextlib.suppress(asyncio.CancelledError):
                await stt_task

    async def _recv_loop(self):
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                await self._on_audio(msg["bytes"])
            elif msg.get("text"):
                await self._on_control(json.loads(msg["text"]))

    async def _on_control(self, msg: dict):
        kind = msg.get("type")
        if kind == "interrupt":
            await self._interrupt()
            await self.set_state(State.IDLE)
        elif kind == "reset":
            await self._interrupt()
            self.history.clear()
            self.vad.reset()
            await self.set_state(State.IDLE)
            await self.send_json({"type": "reset_done"})
        elif kind == "text":  # typed message, skips STT
            text = str(msg.get("text", "")).strip()
            if text:
                await self._interrupt()
                await self._start_reply(text)

    async def _on_audio(self, data: bytes):
        pcm = np.frombuffer(data, dtype=np.int16)
        for ev in self.vad.feed(pcm):
            if ev.kind == "start":
                await self.send_json({"type": "vad", "speaking": True})
                if self.state in (State.THINKING, State.SPEAKING) and self.svc.settings.barge_in:
                    await self._interrupt()
                await self.set_state(State.LISTENING)
            elif ev.kind == "end":
                await self.send_json({"type": "vad", "speaking": False})
                if self.utterances.full():
                    self.utterances.get_nowait()
                self.utterances.put_nowait(ev.audio)
                await self.set_state(State.THINKING)
            else:  # discard: too short to be speech
                await self.send_json({"type": "vad", "speaking": False})
                if self.state == State.LISTENING:
                    await self.set_state(State.IDLE)

    # ---------- STT ----------

    async def _stt_loop(self):
        while True:
            audio = await self.utterances.get()
            t = time.perf_counter()
            try:
                text = await self.svc.stt.transcribe(audio)
            except Exception as e:
                log.exception("STT failed")
                await self.send_json({"type": "error", "msg": f"STT failed: {e}"})
                continue
            stt_ms = int((time.perf_counter() - t) * 1000)
            self._save_debug(audio, text)
            if not text:
                if self.state == State.THINKING:
                    await self.set_state(State.IDLE)
                continue
            await self.send_json({"type": "transcript", "text": text, "stt_ms": stt_ms})
            await self._start_reply(text)

    def _save_debug(self, audio: np.ndarray, text: str):
        folder = self.svc.settings.debug_audio_dir
        if not folder:
            return
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        stem = path / time.strftime("%Y%m%d-%H%M%S")
        sf.write(f"{stem}.wav", audio, 16000)
        Path(f"{stem}.txt").write_text(text + "\n")
        log.info("saved %s.wav (%.1fs): %r", stem, len(audio) / 16000, text)

    # ---------- reply (LLM + TTS) ----------

    async def _start_reply(self, text: str):
        await self._cancel_reply()
        self.turn += 1
        self.reply_task = asyncio.create_task(self._reply(text, self.turn))

    async def _cancel_reply(self) -> bool:
        task, self.reply_task = self.reply_task, None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            return True
        return False

    async def _interrupt(self):
        if await self._cancel_reply():
            await self.send_json({"type": "interrupt"})

    async def _reply(self, text: str, turn: int):
        await self.set_state(State.THINKING)
        self.history.add_user(text)
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        full: list[str] = []
        spoken: list[str] = []
        t0 = time.perf_counter()
        timings: dict[str, int] = {}

        async def think():
            chunker = SentenceChunker()
            try:
                async with self.svc.llm_lock:
                    async for token in self.svc.llm.stream(self.history.messages()):
                        if "first_token_ms" not in timings:
                            timings["first_token_ms"] = int((time.perf_counter() - t0) * 1000)
                        full.append(token)
                        await self.send_json({"type": "llm_token", "turn": turn, "text": token})
                        for s in chunker.push(token):
                            if "first_chunk_ms" not in timings:
                                timings["first_chunk_ms"] = int((time.perf_counter() - t0) * 1000)
                            await sentences.put(s)
                for s in chunker.flush():
                    await sentences.put(s)
            finally:
                await sentences.put(None)

        async def speak():
            rate = self.svc.tts.sample_rate
            step = rate * AUDIO_CHUNK_MS // 1000
            idx = 0
            while (sentence := await sentences.get()) is not None:
                ts = time.perf_counter()
                audio = await self.svc.tts.synth(sentence)
                if not len(audio):
                    continue
                if "first_audio_ms" not in timings:
                    timings["first_tts_ms"] = int((time.perf_counter() - ts) * 1000)
                    timings["first_chunk_text"] = sentence
                    timings["first_audio_ms"] = int((time.perf_counter() - t0) * 1000)
                    await self.set_state(State.SPEAKING)
                await self.send_json(
                    {"type": "tts_start", "turn": turn, "id": idx, "text": sentence, "sample_rate": rate}
                )
                for i in range(0, len(audio), step):
                    await self.send_bytes(audio[i : i + step].tobytes())
                await self.send_json({"type": "tts_end", "turn": turn, "id": idx})
                spoken.append(sentence)
                idx += 1

        try:
            await asyncio.gather(think(), speak())
            self.history.add_assistant("".join(full).strip())
            await self.send_json({"type": "reply_done", "turn": turn, "timings": {k: v for k, v in timings.items() if k != "first_chunk_text"}, "first_chunk": timings.get("first_chunk_text")})
            await self.set_state(State.IDLE)
        except asyncio.CancelledError:
            # Barge-in: remember only what was already sent to the speaker.
            self.history.add_assistant(" ".join(spoken))
            raise
        except Exception as e:
            log.exception("Reply failed")
            await self.send_json({"type": "error", "msg": str(e)})
            await self.set_state(State.IDLE)
