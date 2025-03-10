import logging
import re
from math import log
from pathlib import Path

import jiwer
import numpy as np
from discrete_speech_metrics import PESQ, UTMOS, LogF0RMSE
from pydub import AudioSegment, effects
from resemblyzer import VoiceEncoder, preprocess_wav

from DiscreteSpeechMetrics.discrete_speech_metrics.mcd import MCD

SR = 16_000

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("transcription.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

utmos = UTMOS(sr=SR)
pesq = PESQ(sr=SR)
log_f0_rmse = LogF0RMSE(sr=SR)
mcd = MCD(sr=SR)

encoder = VoiceEncoder()


def normalize_text(text: str) -> str:
    chars_to_ignore_regex = '[,?.!\-\;\:"“%‘”�—’…–„]'

    # Lowercase text and remove special characters
    text = re.sub(chars_to_ignore_regex, "", text.lower())

    # In addition, we can normalize the target text, e.g. removing new lines characters etc...
    token_sequences_to_ignore = ["\n\n", "\n", "   ", "  "]

    for t in token_sequences_to_ignore:
        text = " ".join(text.split(t))
    # logger.debug(f"Normalized text: {text}")
    return text


def compute_wer(reference_text: str, synthesized_text: str) -> float:
    reference_text = normalize_text(reference_text)
    synthesized_text = normalize_text(synthesized_text)

    wer = jiwer.wer(reference_text, synthesized_text)
    return wer


def compute_cer(reference_text: str, synthesized_text: str) -> float:
    reference_text = normalize_text(reference_text)
    synthesized_text = normalize_text(synthesized_text)

    cer = jiwer.cer(reference_text, synthesized_text)
    return cer


def compute_utmos(audio_file):
    return utmos.score(audio_file)


def compute_pesq(reference_audio, synthesized_audio):
    return pesq.score(reference_audio, synthesized_audio)


def compute_log_f0_rmse(reference_audio, synthesized_audio):
    return log_f0_rmse.score(reference_audio, synthesized_audio)


def compute_mcd(reference_audio, synthesized_audio):
    return mcd.score(reference_audio, synthesized_audio)


def load_wav(path: str, normalize=True, target_dbfs=-27):
    # if normalize:
    #     sound = AudioSegment.from_file(path)
    #     change_in_dBFS = target_dbfs - sound.dBFS
    #     normalized_sound = sound.apply_gain(change_in_dBFS)
    #     normalized_sound
    file_path = Path(path)
    wav = preprocess_wav(file_path, source_sr=SR)
    return wav


def cosine_similarity(x, y):
    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def compute_emb(path, normalize=True, target_dbfs=-27):  # target_dbfs=-24.632442475923607
    if normalize:
        song = AudioSegment.from_file(path)
        change_in_dBFS = target_dbfs - song.dBFS
        normalized_sound = song.apply_gain(change_in_dBFS)
        normalized_sound.export(path, format=path[-3:])

    fpath = Path(path)
    wav = preprocess_wav(fpath)
    embed = encoder.embed_utterance(wav)
    return embed


def main():
    pass


if __name__ == "__main__":
    main()
