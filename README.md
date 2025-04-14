# Tinylsa

A Python package for lead sheet analysis and chroma feature extraction.

## Installation

```bash
pip install -r requirements.txt
```

## DeepChroma CLI

The DeepChroma CLI allows you to extract chroma features from audio files using a pre-trained model.

### Basic Usage

```bash
python -m src.deepchroma --audio path/to/audio.wav --model path/to/checkpoint.pth
```

### Options

- `--audio`: Path to the input audio file (required)
- `--model`: Path to the model checkpoint file (required)
- `--output`: Path to save the output features (optional)
- `--patch-size`: Size of patches for inference (default: 128)
- `--hop-size`: Hop size between patches (default: 64)
- `--hidden-dim`: Hidden dimension of the model (default: 128)


### Examples

1. Basic feature extraction:
```bash
python -m src.deepchroma --audio input.wav --model checkpoint.pth
```

2. Save features to file:
```bash
python -m src.deepchroma --audio input.wav --model checkpoint.pth --output features.npy
```

3. Custom patch and hop sizes:
```bash
python -m src.deepchroma --audio input.wav --model checkpoint.pth --patch-size 256 --hop-size 128
```

### Output

If `--output` is specified, the features will be saved as a NumPy array to the specified path.