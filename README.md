# TinyLSA: Lead Sheet Analysis Tool

A tool for aligning audio recordings with lead sheet annotations using deep learning-based chroma features.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tinylsa.git
cd tinylsa

# Install dependencies
pip install -r requirements.txt
```

## Usage

The main tool is the alignment script that matches audio recordings with lead sheet annotations:

```bash
python -m src.align --data <data_folder> --tune <tune_name> --musicxml <musicxml_folder> [options]
```

### Required Arguments

- `--data <data_folder>`: Path to the data folder containing:
  - `audio/`: Folder with audio files (WAV format)
  - `features/`: Folder with beat tracking results (JSON format)
  - `predictions/`: Folder for alignment results (created automatically)
- `--tune <tune_name>`: Name of the tune to process
- `--musicxml <musicxml_folder>`: Path to folder containing MusicXML files

### Optional Arguments

- `--start <seconds>`: Start time in seconds (default: beginning of audio)
- `--end <seconds>`: End time in seconds (default: end of audio)
- `--transpose <0-11>`: Transposition key (0-11, default: try all and select best)
- `--force-chroma-calculation`: Force recalculation of chroma features (default: use cached if available)

### Example

```bash
# Process a 30-90 second segment of jazz standard with transposition 0
python -m src.align \
    --data <data_path> \
    --tune <tune_name> \
    --musicxml <musicxml_path> \
    --transpose 0 \
    --start 30 \
    --end 90

# Process the same tune but force chroma feature recalculation
python -m src.align \
    --data <data_path> \
    --tune <tune_name> \
    --musicxml <musicxml_path> \
    --transpose 0 \
    --start 30 \
    --end 90 \
    --force-chroma-calculation
```

### Output Files

The tool generates the following files:

1. In `features/` folder:
   - `<tune_id>_chroma.npy`: Beat-synchronous chroma features
     - Cached for reuse unless `--force-chroma-calculation` is specified
     - Shape: (num_beats, 12)

2. In `predictions/` folder:
   - `<tune_id>_alignment.json`: Alignment results containing:
     - `loglikelihood`: Log-likelihood of the alignment
     - `states`: Sequence of decoded states
     - `timestamps`: Beat times corresponding to each state

### Data Structure

The expected data structure is:

```
data_folder/
├── audio/
│   └── <tune_id>.wav
├── features/
│   ├── <tune_id>_beats_.json
│   └── <tune_id>_chroma.npy
└── predictions/
    └── <tune_id>_alignment.json
```

The beats JSON file should contain:
- `est_beats`: String of beat frame indices
- `globals`: Dictionary with:
  - `sample_rate`: Audio sample rate
  - `hop_length`: Hop length for beat tracking

## Features

- Beat-synchronous chroma feature extraction
- Automatic transposition detection
- Time segment selection
- Feature caching for efficiency
- JSON-based output format

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- librosa
- music21