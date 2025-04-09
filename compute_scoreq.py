import argparse
import csv
import logging
import os
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

import torchaudio.functional as F

import matplotlib.pyplot as plt
from pathlib import Path


from utils import parse_filelist

SAVE_DIR = "results"
objective_model = SQUIM_OBJECTIVE.get_model()
subjective_model = SQUIM_SUBJECTIVE.get_model()


def plot(waveform, title, sample_rate=16000):
    wav_numpy = waveform.numpy()

    sample_size = waveform.shape[1]
    time_axis = torch.arange(0, sample_size) / sample_rate

    figure, axes = plt.subplots(2, 1)
    axes[0].plot(time_axis, wav_numpy[0], linewidth=1)
    axes[0].grid(True)
    axes[1].specgram(wav_numpy[0], Fs=sample_rate)
    figure.suptitle(title)
    # Save the figure as an image file
    figure.savefig(f"{title}.png")


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
        "PESQ",
        "STOI",
        "SI-SDR",
    ]

    AUDIO_FILE_IDX = results_definition.index("Audio_File")
    PESQ_IDX = results_definition.index("PESQ")
    STOI_IDX = results_definition.index("STOI")
    SI_SDR_IDX = results_definition.index("SI-SDR")

    results_list = [None] * len(results_definition)
    logger.debug(f"Results definition: {results_definition}")
    logger.debug(f"Results list: {results_list}")

    file_exists = os.path.isfile(os.path.join(SAVE_DIR, args.output_csv))

    if file_exists:
        logger.warning(f"File {args.output_csv} already exists ... deleting it")
        os.remove(os.path.join(SAVE_DIR, args.output_csv))

    for idx, line in enumerate(tqdm(filelist, desc="Processing files")):
        # # Skip header
        if idx == 0:
            continue

        base_file, transcript_original, transcript_synthesized = line[0], line[1], line[2]

        audio_path_synthesized = Path(os.path.join(args.base_dir_synthesized, base_file))

        waveform_synthesized, sample_rate = torchaudio.load(audio_path_synthesized)
        if sample_rate != SQUIM_OBJECTIVE.sample_rate:
            logger.debug(f"Resampling from {sample_rate} to {SQUIM_OBJECTIVE.sample_rate}")
            waveform_synthesized = F.resample(waveform_synthesized, sample_rate, SQUIM_OBJECTIVE.sample_rate)

        stoi, pesq, si_sdr = objective_model(waveform_synthesized[:1, :])

        results_list[AUDIO_FILE_IDX] = audio_path_synthesized
        results_list[PESQ_IDX] = pesq[0].item()
        results_list[STOI_IDX] = stoi[0].item()
        results_list[SI_SDR_IDX] = si_sdr[0].item()

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

    with open(args.report_file, "w") as report_file:
        report_file.write(str(desc) + "\n\n\n")

        report_file.write(f"PESQ: {desc['PESQ']['mean']:.3f} $\pm$ {desc['PESQ']['std']:.3f}\n")
        report_file.write(f"STOI: {desc['STOI']['mean']:.3f} $\pm$ {desc['STOI']['std']:.3f}\n")
        report_file.write(f"SI-SDR: {desc['SI-SDR']['mean']:.3f} $\pm$ {desc['SI-SDR']['std']:.3f}\n")


# sample = Path('/workspace/local/samples/matcha-tts-bas950-100.ckpt/bas_rnd2_460.wav')
# reference =  Path('/datasets/SWARA/SWARA1.0_22k_noSil/sgs_rnd2_460.wav')

# waveform_sample, sample_rate = torchaudio.load(sample)
# waveform_sample = torchaudio.functional.resample(waveform_sample, sample_rate, SQUIM_OBJECTIVE.sample_rate)

# if sample_rate != SQUIM_OBJECTIVE.sample_rate:
#     print(f"Resampling from {sample_rate} to {SQUIM_OBJECTIVE.sample_rate}")
#     waveform_sample = torchaudio.functional.resample(waveform_sample, sample_rate, SQUIM_OBJECTIVE.sample_rate)

# print(f"Loaded sample: {sample} with sample rate: {sample_rate}")
# print(waveform_sample.shape)

# waveform_reference, sample_rate = torchaudio.load(reference)
# waveform_reference = torchaudio.functional.resample(waveform_reference, sample_rate, SQUIM_OBJECTIVE.sample_rate)

# if sample_rate != SQUIM_OBJECTIVE.sample_rate:
#     print(f"Resampling from {sample_rate} to {SQUIM_OBJECTIVE.sample_rate}")
#     waveform_reference = torchaudio.functional.resample(waveform_reference, sample_rate, SQUIM_OBJECTIVE.sample_rate)

# print(f"Loaded reference: {reference} with sample rate: {sample_rate}")
# print(waveform_reference.shape)


# # Trim so they have the same number of frames
# if waveform_sample.shape[1] < waveform_reference.shape[1]:
#     waveform_reference = waveform_reference[:, :waveform_sample.shape[1]]
# else:
#     waveform_sample = waveform_sample[:, :waveform_reference.shape[1]]


# plot(waveform_sample, "Sample Speech")


# stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(waveform_sample[:1, :])


# print("Estimated Objective Scores:")

# print(f"STOI: {stoi_hyp[0]}")
# print(f"PESQ: {pesq_hyp[0]}")
# print(f"SI-SDR: {si_sdr_hyp[0]}\n")

# exit(0)






# scores = objective_model(waveform_sample)

# print(f"STOI: {scores[0].item()},  PESQ: {scores[1].item()}, SI-SDR: {scores[2].item()}.")




# waveform_reference = torchaudio.functional.resample(waveform_reference, sample_rate, SQUIM_SUBJECTIVE.sample_rate)
# waveform_sample = torchaudio.functional.resample(waveform_sample, sample_rate, SQUIM_SUBJECTIVE.sample_rate)

# score = subjective_model(waveform_sample, waveform_reference)
# print(f"MOS: {score}.")

# print(f"Estimated MOS for distorted speech at {snr_dbs[0]}dB is MOS: {mos[0]}")

# plot(waveform, "Clean Speech")

# # Predict quality of natural speech in NR mode
# nr_scoreq = scoreq.Scoreq(data_domain='natural', mode='nr')

# # Predict quality of natural speech in REF mode
# ref_scoreq = scoreq.Scoreq(data_domain='natural', mode='ref')

# # Predict quality of synthetic speech in NR mode
# nr_scoreq = scoreq.Scoreq(data_domain='synthetic', mode='nr')

# # Predict quality of synthetic speech in REF mode
# ref_scoreq = scoreq.Scoreq(data_domain='synthetic', mode='ref')

# test_path = '/workspace/local/Matcha-TTS/synth_output/matcha-tts-bas950-60.ckpt/bas_rnd1_001.wav'
# ref_path = '/datasets/SWARA/SWARA1.0_22k_noSil/bas_rnd1_001.wav'

# pred_mos = nr_scoreq.predict(test_path=test_path, ref_path=None)
# print(pred_mos)
# pred_distance = ref_scoreq.predict(test_path=test_path, ref_path=ref_path)
# print(pred_distance)
# pred_mos = nr_scoreq.predict(test_path='./data/opus.wav', ref_path=None)
# print(pred_mos)
# pred_distance = ref_scoreq.predict(test_path='./data/opus.wav', ref_path='./data/ref.wav')
# print(pred_distance)
