import asyncio
import gc

import torch
import whisperx
from config.settings import get_settings


def release_resources(model: object) -> None:
    """
    Run if low on GPU resources. Force garbage collection and release CUDA cache.

    :param model: NN model
    """

    gc.collect()
    torch.cuda.empty_cache()
    del model


async def main() -> None:
    config = get_settings()

    model_asr = whisperx.load_model(
        config.whisper.architecture,
        config.device,
        compute_type=config.whisper.compute_type,
    )

    audio = whisperx.load_audio(config.audio_file)
    result = model_asr.transcribe(audio, batch_size=config.whisper.batch_size)

    release_resources(model_asr)

    model_alignment, metadata = whisperx.load_align_model(language_code=result["language"], device=config.device)
    result = whisperx.align(
        result["segments"],
        model_alignment,
        metadata,
        audio,
        config.device,
        return_char_alignments=False,
    )

    release_resources(model_alignment)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=config.hf_token, device=config.device)

    diarize_segments = diarize_model(audio)
    diarize_model(
        audio,
        min_speakers=config.speaker_diarization.min_speakers,
        max_speakers=config.speaker_diarization.max_speakers,
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(result["segments"])


if __name__ == "__main__":
    asyncio.run(main())
