"""End-to-end check without a microphone.

Speaks a sentence with macOS `say`, streams it to the server like a browser would,
prints every event, and saves the agent's spoken reply to reply.wav.

    uv run python tests/simulate.py "What is the capital of France?"
    uv run python tests/simulate.py --barge-in "Tell me a long story about a dragon."
"""

import argparse
import asyncio
import json
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import websockets

FRAME = 320  # 20 ms at 16 kHz


def speak_16k(text: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run(
            ["say", "-o", f.name, "--file-format=WAVE", "--data-format=LEI16@16000", "--", text], check=True
        )
        audio, _ = sf.read(f.name, dtype="int16")
    return audio


async def stream(ws, pcm: np.ndarray, realtime: bool):
    for i in range(0, len(pcm), FRAME):
        await ws.send(pcm[i : i + FRAME].tobytes())
        if realtime:
            await asyncio.sleep(0.02)


async def main(text: str, url: str, barge_in: bool, out: str):
    speech = speak_16k(text)
    silence = np.zeros(16000, dtype=np.int16)  # 1 s, enough to end the utterance
    audio_out: list[bytes] = []
    rate = 24000
    t0 = None

    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        print("ready:", ready)
        rate = ready["output_sample_rate"]

        await stream(ws, np.concatenate([silence[:4000], speech, silence]), realtime=False)
        t0 = time.perf_counter()
        interrupted = False
        tokens = []

        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=120)
            if isinstance(msg, bytes):
                audio_out.append(msg)
                continue
            ev = json.loads(msg)
            ms = int((time.perf_counter() - t0) * 1000)
            if ev["type"] == "llm_token":
                tokens.append(ev["text"])
                continue
            print(f"+{ms:5d}ms {ev}")
            if ev["type"] == "tts_start" and barge_in and not interrupted and ev["id"] == 0:
                print("        -> user talks over the agent")
                await stream(ws, np.concatenate([speak_16k("Stop. What is two plus two?"), silence]), False)
                interrupted = True
            if ev["type"] == "reply_done" and (not barge_in or interrupted):
                if not barge_in or ev["turn"] >= 2:
                    break
            if ev["type"] == "error":
                break

        print("reply text:", "".join(tokens))

    if audio_out:
        pcm = np.frombuffer(b"".join(audio_out), dtype=np.int16)
        sf.write(out, pcm, rate)
        print(f"saved {len(pcm) / rate:.1f}s of reply audio to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", default="Hi! What is the capital of France?")
    p.add_argument("--url", default="ws://localhost:8000/ws/voice")
    p.add_argument("--barge-in", action="store_true")
    p.add_argument("--out", default="reply.wav")
    a = p.parse_args()
    asyncio.run(main(a.text, a.url, a.barge_in, a.out))
