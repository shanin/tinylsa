"""
Alignment module for matching audio features with lead sheet annotations.
"""

import argparse
import json
import os
from pathlib import Path
import torch
import numpy as np
from typing import Optional, Tuple

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from deepchroma import compute_beat_synchronous_chroma, ChromaInference, ChromaPredictor, HCQTConformer, AudioTrack
from crema import BeatCrema
from leadsheet import LeadSheet

def load_audio_features(audio_path: str, beats_path: str, start_sec: Optional[float] = None, 
                       end_sec: Optional[float] = None, force_recalculation: bool = False, use_crema: bool = True) -> Tuple[torch.Tensor, float, float]:
    """
    Load and process audio features with optional time segment selection.
    
    Args:
        audio_path: Path to audio file
        beats_path: Path to beats JSON file
        start_sec: Optional start time in seconds
        end_sec: Optional end time in seconds
        force_recalculation: If True, recalculate features even if cached file exists
        
    Returns:
        Tuple of (features, actual_start, actual_end)
    """
    # Load beats data
    with open(beats_path, 'r') as f:
        beats_data = json.load(f)
        beats = beats_data['est_beats']
        audio_sr = beats_data['globals']['sample_rate']
        hop_length = beats_data['globals']['hop_length']
    
    print(f"Audio sample rate: {audio_sr}, hop length: {hop_length}")
    print(f"Beats string length: {len(beats)}")
    
    # Parse beat frames
    beat_frames = [int(frame) for frame in beats.strip('[]').split(',')]
    print(f"Number of beats: {len(beat_frames)}")
    print(f"First beat frame: {beat_frames[0]}, Last beat frame: {beat_frames[-1]}")
    
    # Check for cached features
    data_path = Path(audio_path).parent.parent
    data_id = data_path.name
    features_output_path = data_path / 'features' / f'{data_id}_chroma.npy'
    
    if not force_recalculation and features_output_path.exists():
        print(f"Loading cached features from {features_output_path}")
        features = torch.from_numpy(np.load(features_output_path))
    else:
        print("Calculating features from scratch")
        # Create inference object and process audio
        if use_crema:
            model = BeatCrema()
        else:
            model = ChromaPredictor(
                HCQTConformer(
                    hidden_dim=128,
                    output_dim=12,
                ).float()
            )
        inference = ChromaInference(model)
        
        # Load and process audio
        track = AudioTrack(audio_path=audio_path)
        features = inference(track)
        print(f"Initial features shape: {features.shape}")
        
        # Compute beat-synchronous features
        features = compute_beat_synchronous_chroma(features, beats)
        print(f"Beat-synchronous features shape: {features.shape}")
        
        # Save features for future use
        features_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(features_output_path, features.numpy())
    
    # Convert features to numpy for easier slicing
    features = features.numpy()
    
    # Handle time segment selection
    actual_start = 0.0
    actual_end = beat_frames[-1] * hop_length / audio_sr
    
    if start_sec is not None or end_sec is not None:
        print(f"Requested time segment: {start_sec} to {end_sec} seconds")
        
        # Find beat indices that fall within the time segment
        start_beat_idx = 0
        end_beat_idx = len(beat_frames)
        
        for i, frame in enumerate(beat_frames):
            time = frame * hop_length / audio_sr
            if time >= start_sec and start_beat_idx == 0:
                start_beat_idx = i
            if time > end_sec and end_beat_idx == len(beat_frames):
                end_beat_idx = i
                break
        
        print(f"Beat indices range: {start_beat_idx} to {end_beat_idx}")
        
        # Slice features using beat indices
        features = features[start_beat_idx:end_beat_idx]
        actual_start = beat_frames[start_beat_idx] * hop_length / audio_sr
        actual_end = beat_frames[end_beat_idx-1] * hop_length / audio_sr
        
        print(f"Final features shape: {features.shape}")
        print(f"Actual time segment: {actual_start} to {actual_end} seconds")
    
    return torch.from_numpy(features), actual_start, actual_end

def find_musicxml_file(musicxml_folder: str, tune_name: str) -> str:
    """
    Find the MusicXML file for the given tune name.
    
    Args:
        musicxml_folder: Path to MusicXML folder
        tune_name: Name of the tune
        
    Returns:
        Path to the MusicXML file
    """
    musicxml_folder = Path(musicxml_folder)
    for file in musicxml_folder.glob('*.musicxml'):
        if tune_name.lower() in file.stem.lower():
            return str(file)
    raise FileNotFoundError(f"Could not find MusicXML file for tune '{tune_name}'")

def main():
    parser = argparse.ArgumentParser(description='Align audio features with lead sheet annotations')
    parser.add_argument('--data', type=str, required=True, help='Path to data folder with audio and features subfolders')
    parser.add_argument('--tune', type=str, required=True, help='Name of the tune')
    parser.add_argument('--musicxml', type=str, required=True, help='Path to MusicXML folder')
    parser.add_argument('--start', type=float, help='Start time in seconds')
    parser.add_argument('--end', type=float, help='End time in seconds')
    parser.add_argument('--transpose', type=int, choices=range(12), help='Transposition key (0-11)')
    parser.add_argument('--force-chroma-calculation', action='store_true', help='Force recalculation of chroma features')
    parser.add_argument('--segment-id', type=str, help='Identifier for the segment (used in output filename)')
    parser.add_argument('--crema', action='store_true', help='Use CREMA model')

    args = parser.parse_args()
    
    # Construct paths
    data_path = Path(args.data)
    data_id = data_path.name
    audio_path = data_path / 'audio' / f'{data_id}.wav'
    beats_path = data_path / 'features' / f'{data_id}_beats_.json'
    
    # Create output filename with segment_id if provided
    output_suffix = f"_{args.segment_id}" if args.segment_id else ""
    
    # Load and process audio features
    features, start_time, end_time = load_audio_features(
        str(audio_path), 
        str(beats_path),
        args.start,
        args.end,
        args.force_chroma_calculation,
        args.crema,
    )
    
    # Find and load MusicXML file
    musicxml_path = find_musicxml_file(args.musicxml, args.tune)
    leadsheet = LeadSheet(musicxml_path)
    
    # Handle transposition
    if args.transpose is not None:
        # Decode with specified transposition
        print(features.numpy().shape)
        loglik, states = leadsheet.decode(features.numpy(), transposition=args.transpose)
    else:
        # Try all transpositions and select best
        best_loglik = float('-inf')
        best_states = None
        
        for transposition in range(12):
            loglik, states = leadsheet.decode(features.numpy(), transposition=transposition)
            if loglik > best_loglik:
                best_loglik = loglik
                best_states = states
        
        loglik, states = best_loglik, best_states
    
    # Load beats data for timestamps
    with open(beats_path, 'r') as f:
        beats_data = json.load(f)
        beats = beats_data['est_beats']
        audio_sr = beats_data['globals']['sample_rate']
        hop_length = beats_data['globals']['hop_length']
    
    # Parse beat frames and convert to timestamps
    beat_frames = [int(frame) for frame in beats.strip('[]').split(',')]
    beat_times = [float(frame) * hop_length / audio_sr for frame in beat_frames]
    
    # Find beat indices that fall within the time segment
    start_beat_idx = 0
    end_beat_idx = len(beat_frames)
    
    if args.start is not None or args.end is not None:
        for i, time in enumerate(beat_times):
            if time >= args.start and start_beat_idx == 0:
                start_beat_idx = i
            if time > args.end and end_beat_idx == len(beat_frames):
                end_beat_idx = i
                break
    
    # Get only the timestamps for our segment
    segment_beat_times = beat_times[start_beat_idx:end_beat_idx]
    
    # Create alignment result with timestamps
    alignment_result = {
        'loglikelihood': float(loglik),  # Convert to Python float
        'states': [int(state) for state in states],  # Convert to Python int
        'timestamps': segment_beat_times  # Only timestamps for our segment
    }
    
    # Create predictions directory if it doesn't exist
    predictions_path = data_path / 'predictions'
    predictions_path.mkdir(parents=True, exist_ok=True)
    
    # Save alignment result with segment_id in filename
    result_path = predictions_path / f'{data_id}_alignment{output_suffix}.json'
    with open(result_path, 'w') as f:
        json.dump(alignment_result, f, indent=2)
    
    print(f"Alignment saved to {result_path}")

if __name__ == '__main__':
    main()
