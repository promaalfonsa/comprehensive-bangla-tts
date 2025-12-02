# Data Directory for Bangla TTS Voice Training

This directory contains recorded audio files for training a custom Bangla VITS TTS voice.

## Directory Structure

```
data/
└── wavs/           # WAV audio files (16-bit PCM, 22050 Hz)
    ├── 1.wav       # Recording for sentence #1
    ├── 2.wav       # Recording for sentence #2
    └── ...
```

## Audio Format

- **Format**: WAV (16-bit PCM)
- **Sample Rate**: 22050 Hz
- **Channels**: Mono (1 channel)

## Usage

Audio files are automatically named based on the sentence ID from `training sentence data.csv`.
Use the Flask recording app at `recording_app/app.py` to record and manage audio files.
