# Audio Dataset Preparation Guide

## Dataset Structure
The dataset should be organized in the following structure:
```
dataset/
├── real/     # Authentic audio samples
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── fake/     # Synthetic/deepfake audio samples
    ├── sample1.wav
    ├── sample2.wav
    └── ...
```

## Requirements
- Audio files should be in WAV or MP3 format
- Recommended sample rate: 22050 Hz
- Recommended duration: 3 seconds
- If audio samples are longer, they will be truncated to the first 3 seconds
- If audio samples are shorter, they will be padded with silence

## Dataset Organization Guidelines
1. **Real Audio Samples (`real/` directory)**
   - Record authentic human speech samples
   - Use high-quality microphones for recording
   - Ensure clean audio without background noise
   - Include various speakers, accents, and content

2. **Fake Audio Samples (`fake/` directory)**
   - Generate synthetic speech using text-to-speech or voice cloning tools
   - Include deepfake audio created with various AI models
   - Ensure variety in synthetic voice characteristics

## Data Processing
The system will automatically:
- Convert audio to mel spectrograms
- Normalize the features
- Split the dataset into training and testing sets (80/20 split)

## Usage
1. Create the dataset directory structure:
```bash
mkdir -p dataset/real dataset/fake
```

2. Place your audio files in the respective directories:
   - Authentic audio files in `dataset/real/`
   - Synthetic/deepfake audio files in `dataset/fake/`

3. Update the `data_dir` path in `main.py` to point to your dataset directory

## Recommended Tools for Data Collection
- For real samples:
  - Professional audio recording equipment
  - Voice recording apps with high-quality settings

- For fake samples:
  - Text-to-speech services
  - Voice conversion models
  - Audio synthesis tools

Note: Ensure you have proper rights and permissions for all audio samples used in the dataset.