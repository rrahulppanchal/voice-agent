"""Streaming voice activity detection: turns a stream of PCM frames into utterances."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
FRAME = 512  # Silero expects 512-sample windows at 16 kHz (32 ms)

import os as _os
torch.set_num_threads(int(_os.environ.get("VA_TORCH_THREADS", "6")))  # shared by VAD and Kokoro TTS


def load_vad_model():
    """Each session gets its own model instance because Silero keeps internal state."""
    return load_silero_vad()


@dataclass
class VadEvent:
    kind: str  # "start" | "end" | "discard"
    audio: np.ndarray | None = None  # int16 utterance on "end"


class VADSegmenter:
    def __init__(
        self,
        model,
        threshold: float,
        min_silence_ms: int,
        speech_pad_ms: int,
        min_speech_ms: int,
        max_speech_s: float,
    ):
        self.model = model
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.05)
        self.min_silence = int(min_silence_ms * SAMPLE_RATE / 1000)
        self.min_speech = int(min_speech_ms * SAMPLE_RATE / 1000)
        self.max_speech = int(max_speech_s * SAMPLE_RATE)
        self.pre = deque(maxlen=max(1, int(speech_pad_ms * SAMPLE_RATE / 1000) // FRAME))
        self.buf = np.zeros(0, dtype=np.int16)
        self.reset()

    def reset(self):
        self.triggered = False
        self.speech: list[np.ndarray] = []
        self.speech_len = 0
        self.silence = 0
        self.pre.clear()
        self.model.reset_states()

    def _prob(self, frame: np.ndarray) -> float:
        x = torch.from_numpy(frame.astype(np.float32) / 32768.0)
        with torch.no_grad():
            return float(self.model(x, SAMPLE_RATE).item())

    def feed(self, pcm: np.ndarray) -> list[VadEvent]:
        events: list[VadEvent] = []
        self.buf = np.concatenate([self.buf, pcm])
        while len(self.buf) >= FRAME:
            frame, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            prob = self._prob(frame)

            if not self.triggered:
                self.pre.append(frame)
                if prob >= self.threshold:
                    self.triggered = True
                    self.speech = list(self.pre)
                    self.speech_len = sum(len(f) for f in self.speech)
                    self.silence = 0
                    events.append(VadEvent("start"))
                continue

            self.speech.append(frame)
            self.speech_len += len(frame)
            self.silence = self.silence + len(frame) if prob < self.neg_threshold else 0

            if self.silence >= self.min_silence or self.speech_len >= self.max_speech:
                audio = np.concatenate(self.speech)
                voiced = self.speech_len - self.silence
                self.reset()
                if voiced >= self.min_speech:
                    events.append(VadEvent("end", audio))
                else:
                    events.append(VadEvent("discard"))
        return events
