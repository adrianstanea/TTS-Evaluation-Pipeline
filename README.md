# TTS-Evaluation-Pipeline

Sample scripts used during the evaluation of various TTS Generators adapted for the Romanian Language

## Speech Synthesis

- Using the TTS generator, we will generate speech from text with the finetuned models (SGS and BAS in our case).
- The synthesized speech for each fine-tuned model is stored within a folder with the name of the speaker.

## Transcription

- Paths to the original audio files using metadata of our dataset.
- For the synthesized speech we only need the base_dir path since the speech files have the same name.
- Generate transcriptions for the Romanian language using the Whisper model.
- This will generate a csv with:
  - Original audio file.
  - Transcription of the original audio file.
  - Transcription of the synthesized speech.

- This data allows us to compute text metrics. Also for future usage we only need the base dir for original and synthesized speech.

- Example usage for **transcription.py**:

```bash
python3 transcription.py \
    --input_filelist /home/astanea/git-repos/TTS/Speech-Backbones/Grad-TTS/out/finetune_meta_bas_1490_samples.csv \
    --base_dir_synthesized /home/astanea/git-repos/TTS/Speech-Backbones/Grad-TTS/out/bas \
    --generator_model grad-tts-BAS
```

## Metrics

- Using the .csv from the transcription, we have access to the original and synthesized transcriptions along with paths for both original and synthesized speech.
- The script generates another .csv with a set of metrics computed for each individual audio file which allows uus to compute various statistics.

- Example usage for **metrics.py**:

```bash
python3 metrics.py
    --input_filelist /home/astanea/git-repos/TTS/evaluation/matcha-tts-BAS.csv  \
    --base_dir_synthesized /home/astanea/git-repos/TTS/Matcha-TTS/synth_output/bas/ \
    --base_dir_original /datasets/SWARA/SWARA1.0_22k_noSil \
    --output_csv ./results/metrics_grad-tts-BAS.csv
```
