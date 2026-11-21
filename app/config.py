from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables. Override with env vars prefixed VA_ (e.g. VA_LLM_MODEL=qwen3:4b) or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VA_")

    # LLM (Ollama)
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:4b-instruct"  # fastest; qwen3:8b is smarter but ~2x slower
    llm_max_tokens: int = 200
    llm_keep_alive: str = "30m"
    llm_num_ctx: int = 4096  # small context = less memory, faster prompt processing
    system_prompt: str = (
        "You are a friendly, helpful voice assistant. Your replies are spoken aloud, so answer "
        "in one to three short, natural sentences. Never use markdown, lists, code, or emojis."
    )
    max_history_turns: int = 10

    # Speech-to-text
    stt_engine: str = "mlx"  # mlx (Apple GPU) | faster-whisper (CPU)
    stt_model: str = "small.en-mlx"  # mlx: small.en-mlx (best here) | large-v3-turbo (multilingual); faster-whisper: base.en | small.en
    stt_language: str = "en"  # "" = auto-detect any language (use a non-.en model)
    stt_prompt: str = ""  # vocabulary hint: names, places, product terms you say often
    stt_beam_size: int = 1  # faster-whisper only: 5 = more accurate, slower
    stt_compute_type: str = "int8"  # faster-whisper only
    debug_audio_dir: str = ""  # e.g. "debug_audio": save every utterance as WAV + transcript

    # Text-to-speech
    tts_engine: str = "auto"  # auto | kokoro | say
    tts_voice: str = "af_heart"  # Kokoro voice
    tts_device: str = "cpu"  # cpu | mps (Apple GPU)
    say_voice: str = ""  # macOS `say` voice, empty = system default

    # Voice activity detection (Silero)
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 600  # silence that ends an utterance
    vad_speech_pad_ms: int = 300  # audio kept from before speech starts
    vad_min_speech_ms: int = 250  # shorter blips are ignored
    vad_max_speech_s: float = 30.0  # force-end very long utterances

    # Behaviour
    barge_in: bool = True  # user speech interrupts the agent


settings = Settings()
