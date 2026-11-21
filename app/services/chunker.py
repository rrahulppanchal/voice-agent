"""Turns a token stream into speakable chunks, so TTS can start before the LLM finishes.

Latency trick: the first chunks are cut early (a few words, or at the first comma) so audio
starts within a few hundred ms. Later chunks are whole sentences for natural prosody, and are
ready long before the earlier audio finishes playing.
"""

import re

_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*(?=\s)")
_SOFT_BREAK = re.compile(r"[,;:—–]\s|\s-\s")
_MARKDOWN = re.compile(r"[*_#`>~]|\[|\]\([^)]*\)")


def clean_for_speech(text: str) -> str:
    text = _MARKDOWN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _words_before(text: str, end: int) -> int:
    return len(text[:end].split())


class SentenceChunker:
    def __init__(self, early_word_caps: tuple[int, ...] = (3, 10), max_chars: int = 180):
        # early_word_caps[i]: chunk i is cut after this many complete words even without punctuation.
        self.early_word_caps = early_word_caps
        self.max_chars = max_chars
        self.emitted = 0
        self.buf = ""

    def _cut_point(self) -> int | None:
        buf = self.buf
        m = _SENTENCE_END.search(buf)
        if m:
            return m.end()

        if self.emitted < len(self.early_word_caps):
            cap = self.early_word_caps[self.emitted]
            for s in _SOFT_BREAK.finditer(buf):
                if _words_before(buf, s.end()) >= 2:
                    return s.end()
            # complete words = words followed by whitespace
            complete = [m.end() for m in re.finditer(r"\S+\s", buf)]
            if len(complete) >= cap:
                return complete[cap - 1]
            return None

        if len(buf) > self.max_chars:
            soft = list(_SOFT_BREAK.finditer(buf, 0, self.max_chars))
            cut = soft[-1].end() if soft else buf.rfind(" ", 0, self.max_chars) + 1
            return cut if cut > 0 else self.max_chars
        return None

    def push(self, token: str) -> list[str]:
        self.buf += token
        out: list[str] = []
        while (cut := self._cut_point()) is not None:
            chunk = clean_for_speech(self.buf[:cut])
            self.buf = self.buf[cut:]
            if chunk:
                out.append(chunk)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        chunk = clean_for_speech(self.buf)
        self.buf = ""
        return [chunk] if chunk else []
