import argparse
import csv
import logging
import os

import whisper
from tqdm import tqdm

from utils import parse_filelist

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("transcription.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Setup Whisper
model = whisper.load_model("medium")
language = "ro"
options = dict(language=language, beam_size=5, best_of=5)
transcribe_options = dict(task="transcribe", **options)


def transcribe_audio(audio_path: str) -> str:
    audio = whisper.load_audio(audio_path)
    transcription = model.transcribe(audio, **transcribe_options)["text"]
    return transcription


def save_transcriptions_to_csv(
    audio_file, original_trans, synthesized_trans, output_csv
):
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        # Write headers if file is new
        if not file_exists:
            writer.writerow(
                ["Audio File", "Original Transcription", "Synthesized Transcription"]
            )
        writer.writerow([audio_file, original_trans, synthesized_trans])


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
        "--generator_model",
        type=str,
        required=True,
        help="Path to the generator model.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    filelist = parse_filelist(args.input_filelist, split_char="|")
    for idx, line in enumerate(tqdm(filelist, desc="Processing files")):
        if idx < 352:
            continue

        filepath, _, _ = line[0], line[1], line[2]

        audio_file = os.path.basename(filepath)

        original_audio_file = filepath
        synthesized_audio_file = os.path.join(args.base_dir_synthesized, audio_file)

        trascription_original = transcribe_audio(original_audio_file)
        transcription_synthesized = transcribe_audio(synthesized_audio_file)

        save_transcriptions_to_csv(
            audio_file,
            trascription_original,
            transcription_synthesized,
            output_csv=f"{args.generator_model}.csv",
        )


if __name__ == "__main__":
    main()
