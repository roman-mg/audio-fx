from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class SpeakerDiarization(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPEAKER_DIARIZATION_", env_file=".env", extra="ignore")

    min_speakers: int = 0
    max_speakers: int = 8


class WhisperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WHISPER_", env_file=".env", extra="ignore")

    batch_size: int = 16  # reduce if low on GPU mem
    compute_type: str = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
    architecture: str = "tiny"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    device: str = "cuda"
    audio_file: str = "../resources/audio.mp3"
    hf_token: str = ""
    whisper: WhisperSettings = WhisperSettings()
    speaker_diarization: SpeakerDiarization = SpeakerDiarization()


@lru_cache()
def get_settings() -> Settings:
    return Settings()
