import argparse
import csv
import logging
import os

import pandas as pd
from tqdm import tqdm

from compute_metrics import (
    compute_cer,
    compute_wer,
)
from utils import parse_filelist

SAVE_DIR = "redo_WER_CER"


def save_to_csv(data, header, output_csv):

    file_exists = os.path.isfile(os.path.join(SAVE_DIR, output_csv))

    with open(os.path.join(SAVE_DIR, output_csv), "a", newline="") as f:
        writer = csv.writer(f, delimiter="|")

        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument(
        "--input_filelist",
        type=str,
        # required=True,
        help="Filelist containing the original audio files and their transcriptions. ",
    )
    parser.add_argument(
        "--base_dir_synthesized",
        type=str,
        # required=True,
        help="Base directory of the synthesized audio files. Audio files use the same name as the original audio files.",
    )
    parser.add_argument(
        "--base_dir_original",
        type=str,
        # required=True,
        help="Base directory of the original audio files. Audio files use the same name as the original audio files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        # required=True,
        help="Path to the generator model.",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        # required=True,
        help="Synthesized statistics report file.",
    )
    args = parser.parse_args()
    return args


def main(args=None):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("transcription.log"), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)

    filelist = parse_filelist(args.input_filelist, split_char="|")

    os.makedirs(SAVE_DIR, exist_ok=True)

    results_definition = [
        "Audio_File",
        "Original_Transcription",
        "Synthesized_Transcription",
        "CER",
        "WER",
    ]

    AUDIO_FILE_IDX = results_definition.index("Audio_File")
    ORIGINAL_TRANSCRIPTION_IDX = results_definition.index("Original_Transcription")
    SYNTHESIZED_TRANSCRIPTION_IDX = results_definition.index("Synthesized_Transcription")
    CER_IDX = results_definition.index("CER")
    WER_IDX = results_definition.index("WER")

    results_list = [None] * len(results_definition)
    logger.debug(f"Results definition: {results_definition}")
    logger.debug(f"Results list: {results_list}")

    for idx, line in enumerate(tqdm(filelist, desc="Processing files")):
        # # Skip header
        if idx == 0:
            continue

        base_file, transcript_original, transcript_synthesized = line[0], line[1], line[2]

        # Compute metrics
        cer = compute_cer(transcript_original, transcript_synthesized)
        wer = compute_wer(transcript_original, transcript_synthesized)


        results_list[AUDIO_FILE_IDX] = base_file
        results_list[ORIGINAL_TRANSCRIPTION_IDX] = transcript_original
        results_list[SYNTHESIZED_TRANSCRIPTION_IDX] = transcript_synthesized
        results_list[CER_IDX] = cer
        results_list[WER_IDX] = wer

        save_to_csv(results_list, results_definition, args.output_csv)


if __name__ == "__main__":
    args = parse_args()

    main(args)

    df = pd.read_csv(os.path.join(SAVE_DIR, args.output_csv), delimiter="|", header=0)

    desc = df.describe()
    print(desc)

    if args.report_file is None:
        print("Report file not specified. Exiting.")
        exit(1)

    with open(os.path.join(SAVE_DIR, args.report_file), "w") as report_file:
        report_file.write(str(desc) + "\n\n\n")

        report_file.write(f"CER: {desc['CER']['mean']:.3f} ± {desc['CER']['std']:.3f}\n")
        report_file.write(f"WER: {desc['WER']['mean']:.3f} ± {desc['WER']['std']:.3f}\n")