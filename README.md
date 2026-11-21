# Local Voice Agent (FastAPI)

Talk to a local LLM in your browser. Everything runs on this Mac: no cloud, no API keys.

```
Browser mic ──16 kHz PCM──► /ws/voice ──► Silero VAD ──► faster-whisper ──► Ollama (qwen3:4b-instruct)
Browser speaker ◄──24 kHz PCM── Kokoro TTS ◄── sentence chunker ◄── token stream
```

## Requirements (already set up on this machine)

- Ollama running with the model: `brew services start ollama` and `ollama pull qwen3:4b-instruct`
- `brew install espeak-ng` (Kokoro uses it for pronunciation)
- [uv](https://docs.astral.sh/uv/)

## Run

```bash
uv run uvicorn app.main:app --port 8000
```

Open http://localhost:8000, press **Start**, allow the microphone and talk. Use headphones;
otherwise the agent can hear itself and interrupt its own replies.

First start downloads Whisper and Kokoro weights (~500 MB) and takes a while; later starts take ~15 s.

## Test without a microphone

```bash
uv run python tests/simulate.py "What is the capital of France?"
uv run python tests/simulate.py --barge-in "Tell me a long story about a dragon."
```

This speaks the text with macOS `say`, streams it to the server, prints every event and writes the reply to `reply.wav`.

## Configuration

Set environment variables (or a `.env` file) with the `VA_` prefix. See `app/config.py`.

| Variable | Default | Notes |
|---|---|---|
| `VA_LLM_MODEL` | `qwen3:4b-instruct` | any Ollama model; `qwen3:8b` is smarter but ~2x slower. Avoid `qwen3:4b` (thinking-only) |
| `VA_STT_MODEL` | `base.en` | `tiny.en` faster, `small.en` more accurate |
| `VA_TTS_ENGINE` | `auto` | `kokoro`, `say` (macOS voice) or `auto` |
| `VA_TTS_VOICE` | `af_heart` | Kokoro voice (`am_adam`, `bf_emma`, ...) |
| `VA_VAD_MIN_SILENCE_MS` | `600` | pause that ends your turn; lower = snappier |
| `VA_BARGE_IN` | `true` | talking over the agent interrupts it |
| `VA_TTS_DEVICE` | `cpu` | `mps` (GPU) is slower here: it competes with Ollama |
| `VA_TORCH_THREADS` | `6` | CPU threads for Kokoro + VAD |
| `VA_SYSTEM_PROMPT` | short spoken replies | the agent's personality |

## WebSocket protocol (`/ws/voice`)

Client → server
- binary: int16 mono PCM at 16 kHz (any frame size; the browser sends 20 ms)
- `{"type":"text","text":"..."}`: typed message, skips STT
- `{"type":"interrupt"}`: stop the current reply
- `{"type":"reset"}`: clear the conversation

Server → client
- `ready`: sample rates and model names
- `state`: `idle | listening | thinking | speaking`
- `vad`: `{"speaking": true|false}`
- `transcript`: what you said, with `stt_ms`
- `llm_token`: streamed reply text
- `tts_start` / binary int16 PCM chunks (100 ms) / `tts_end`: one per sentence
- `reply_done`: `timings.first_token_ms`, `timings.first_audio_ms`
- `interrupt`: flush queued audio now
- `error`

## Layout

```
app/
  main.py               FastAPI app, model loading at startup, routes
  config.py             settings (VA_* env vars)
  session/session.py    per-connection pipeline, state machine, barge-in
  session/history.py    trimmed conversation memory
  services/vad.py       streaming Silero VAD -> utterances
  services/stt.py       faster-whisper (thread pool)
  services/llm.py       Ollama streaming client
  services/chunker.py   tokens -> speakable sentences
  services/tts.py       Kokoro TTS, macOS `say` fallback
  services/container.py shared models + LLM lock
static/index.html       browser client (mic capture, playback, transcript)
static/mic-worklet.js   AudioWorklet: resample to 16 kHz int16
tests/simulate.py       end-to-end test without a mic
```

## Limits

- One LLM reply at a time (16 GB RAM); extra users wait in line.
- After an interruption, the history keeps the sentences that were *sent* to the browser, which can be slightly more than you actually heard.
- Swap any stage by writing a class with the same method (`transcribe`, `stream`, `synth`), e.g. a cloud STT or LLM.
