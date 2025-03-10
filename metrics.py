import argparse
import csv
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from compute_metrics import (
    compute_cer,
    compute_emb,
    compute_log_f0_rmse,
    compute_mcd,
    compute_pesq,
    compute_utmos,
    compute_wer,
    cosine_similarity,
    load_wav,
)
from utils import parse_filelist

SAVE_DIR = "results"


def save_to_csv(data, header, output_csv):

    file_exists = os.path.isfile(os.path.join(SAVE_DIR, output_csv))

    with open(os.path.join(SAVE_DIR, output_csv), "a", newline="") as f:
        writer = csv.writer(f, delimiter="|")

        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("transcription.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument(
        "--input_filelist",
        type=str,
        required=True,
        help="Filelist containing the original audio files and their transcriptions. ",
    )
    parser.add_argument(
        "--base_dir_synthesized",
        type=str,
        required=True,
        help="Base directory of the synthesized audio files. Audio files use the same name as the original audio files.",
    )
    parser.add_argument(
        "--base_dir_original",
        type=str,
        required=True,
        help="Base directory of the original audio files. Audio files use the same name as the original audio files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the generator model.",
    )
    args = parser.parse_args()
    return args


def main(args=None):
    filelist = parse_filelist(args.input_filelist, split_char="|")

    os.makedirs(SAVE_DIR, exist_ok=True)

    results_definition = [
        "Audio_File",
        "Original_Transcription",
        "Synthesized_Transcription",
        "CER",
        "WER",
        "UTMOS_Original",
        "UTMOS_Synthesized",
        "Log_F0_RMSE",
        "MCD",
        "Speaker_Cosine_Similarity",
    ]

    AUDIO_FILE_IDX = results_definition.index("Audio_File")
    ORIGINAL_TRANSCRIPTION_IDX = results_definition.index("Original_Transcription")
    SYNTHESIZED_TRANSCRIPTION_IDX = results_definition.index("Synthesized_Transcription")
    CER_IDX = results_definition.index("CER")
    WER_IDX = results_definition.index("WER")
    UTMOS_ORIGINAL_IDX = results_definition.index("UTMOS_Original")
    UTMOS_SYNTHESIZED_IDX = results_definition.index("UTMOS_Synthesized")
    LOG_F0_RMSE_IDX = results_definition.index("Log_F0_RMSE")
    MCD_IDX = results_definition.index("MCD")
    SPEAKER_COSINE_SIMILARITY_IDX = results_definition.index("Speaker_Cosine_Similarity")

    results_list = [None] * len(results_definition)
    logger.debug(f"Results definition: {results_definition}")
    logger.debug(f"Results list: {results_list}")

    for idx, line in enumerate(tqdm(filelist, desc="Processing files")):
        # Skip header
        if idx == 0:
            continue

        base_file, transcript_original, transcript_synthesized = line[0], line[1], line[2]

        audio_path_original = os.path.join(args.base_dir_original, base_file)
        audio_path_synthesized = os.path.join(args.base_dir_synthesized, base_file)

        emb_original = compute_emb(audio_path_original)
        emb_synthesized = compute_emb(audio_path_synthesized)

        wav_original = load_wav(audio_path_original)
        wav_synthesized = load_wav(audio_path_synthesized)

        logger.debug(f"Original: {audio_path_original}")
        logger.debug(f"Synthesized: {audio_path_synthesized}")
        logger.debug(f"Wav original shape: {wav_original.shape}")
        logger.debug(f"Wav synthesized shape: {wav_synthesized.shape}")

        # Compute metrics
        cer = compute_cer(transcript_original, transcript_synthesized)
        wer = compute_wer(transcript_original, transcript_synthesized)

        utmos_original = compute_utmos(wav_original)
        utmos_synthesized = compute_utmos(wav_synthesized)

        log_f0_rmse = compute_log_f0_rmse(wav_original, wav_synthesized)
        mcd = compute_mcd(wav_original, wav_synthesized)

        speaker_cosine_similarity = cosine_similarity(emb_original, emb_synthesized)

        results_list[AUDIO_FILE_IDX] = base_file
        results_list[ORIGINAL_TRANSCRIPTION_IDX] = transcript_original
        results_list[SYNTHESIZED_TRANSCRIPTION_IDX] = transcript_synthesized
        results_list[CER_IDX] = cer
        results_list[WER_IDX] = wer
        results_list[UTMOS_ORIGINAL_IDX] = utmos_original
        results_list[UTMOS_SYNTHESIZED_IDX] = utmos_synthesized
        results_list[LOG_F0_RMSE_IDX] = log_f0_rmse
        results_list[MCD_IDX] = mcd
        results_list[SPEAKER_COSINE_SIMILARITY_IDX] = speaker_cosine_similarity

        save_to_csv(results_list, results_definition, args.output_csv)


if __name__ == "__main__":
    args = parse_args()

    # main(args)

    df = pd.read_csv(args.output_csv, delimiter="|", header=0)
    # df = pd.read_csv(os.path.join(SAVE_DIR, args.output_csv), delimiter="|", header=0)

    print(df.head())
    print(df.describe())
