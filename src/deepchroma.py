import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer
import math
import lightning as L


import os
import librosa

from pumpp.feature import HCQTMag
from pumpp import Pump

import torch
import numpy as np

import argparse
from pathlib import Path


class AudioTrack:

    def load_audio(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file for {self.name} not found")
        audio, sr = librosa.load(audio_path, sr=44100)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, sr

    def load_features(self, features_path):
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file for {self.name} not found")
        return np.load(features_path)

    def sec_to_frame(self, sec):
        return int(sec * self.sr / self.hop_length)

    def generate_features(self, audio, sr=44100, hop_length=4096, n_octaves=6):
        assert audio.ndim == 1
        p_feature = HCQTMag(
            name="cqt",
            sr=sr,
            hop_length=hop_length,
            harmonics=[1, 2],
            log=True,
            conv="tf",
            n_octaves=n_octaves,
        )
        pump = Pump(p_feature)
        features = torch.tensor(pump(y=audio, sr=sr)["cqt/mag"][0])
        num_windows = features.shape[0]
        window = 4096 / 44100
        return features, num_windows, window

    def __init__(
        self,
        audio_path=None,
        features_path=None,
        force_features_calculation=False,
        sr=44100,
        hop=4096,
    ):
        self.sr = sr
        self.hop = hop
        save_features = True
        if features_path is None:
            force_features_calculation = True
            save_features = False
        elif not os.path.exists(features_path):
            force_features_calculation = True
        if force_features_calculation:
            self.audio, self.sr = self.load_audio(audio_path)
            self.features, self.num_windows, self.window = self.generate_features(
                self.audio, self.sr, self.hop
            )
        else:
            self.features = self.load_features(features_path)

        if save_features:
            os.makedirs(os.path.dirname(features_path), exist_ok=True)
            np.save(features_path, self.features)


class AudioDataset:

    def get_file_list(self):
        if self.audio_dir is not None:
            list_dir = os.listdir(self.audio_dir)
            list_dir = [f for f in list_dir if f.endswith(self.audio_ext)]
        elif self.features_dir is not None:
            list_dir = os.listdir(self.features_dir)
            list_dir = [
                f.replace(self.features_ext, self.audio_ext)
                for f in list_dir
                if f.endswith(self.features_ext)
            ]
        return sorted(list_dir)

    def __init__(
        self,
        audio_dir=None,
        features_dir=None,
        audio_ext=".flac",
        features_ext=".npy",
        sr=44100,
        hop=4096,
    ):
        self.audio_dir = audio_dir
        self.features_dir = features_dir
        self.features_ext = features_ext
        self.audio_ext = audio_ext
        self.file_list = self.get_file_list()

    def __len__(self):
        return len(self.file_list)

    def _get_audio_path(self, index):
        if self.audio_dir is not None:
            return self.audio_dir + "/" + self.file_list[index]
        else:
            return None

    def _get_features_path(self, index):
        if self.features_dir is not None:
            return (
                self.features_dir
                + "/"
                + self.file_list[index].replace(self.audio_ext, self.features_ext)
            )
        else:
            return None

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        features_path = self._get_features_path(index)
        return AudioTrack(
            audio_path=audio_path,
            features_path=features_path,
        )


class AudioPatches:
    def __init__(self, audio, patch_size=100, hop_size=50):
        self.X = []

        track = audio

        track_length = track.features.shape[0]
        if track_length < patch_size:
            padding_size = patch_size - track_length
            padding = torch.zeros(padding_size, track.features.shape[1], track.features.shape[2])
            track.features = torch.cat([torch.tensor(track.features), padding], dim=0)
            track_length = track.features.shape[0]
            

        # Process regular patches with hop_size step
        for i in range(0, track_length - patch_size, hop_size):
            self.X.append(track.features[i : i + patch_size].clone().detach())

        # Handle the last patch with padding if needed
        last_start = max(0, track_length - patch_size)
        if last_start < track_length and (
            last_start == 0
            or last_start >= (track_length - patch_size) // hop_size * hop_size
        ):
            # Get last section and pad if necessary
            last_features = track.features[last_start:track_length]

            if last_features.shape[0] < patch_size:
                # Create feature padding (repeat last frame)
                padding_size = patch_size - last_features.shape[0]
                if isinstance(last_features, torch.Tensor):
                    feature_padding = last_features[-1].repeat(padding_size, 1)
                    last_features_padded = torch.cat(
                        [last_features, feature_padding], dim=0
                    )

                else:
                    feature_padding = np.tile(last_features[-1], (padding_size, 1))
                    last_features_padded = np.vstack([last_features, feature_padding])

                self.X.append(last_features_padded.clone().detach())
            else:
                self.X.append(last_features.clone().detach())

        self.X = torch.stack(self.X).float()


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]




class BaseSequence(nn.Module):
    def create_sinusoidal_positional_encoding(self, seq_len, hidden_dim):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(seq_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def add_positional_encoding(self, x):
        seq_len = x.size(1)
        return x + self.position_encoding[:, :seq_len, :].to(x.device)


class HCQTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size=(5, 5), padding="same", dtype=torch.float
        )
        self.bn1 = nn.BatchNorm2d(1)

    def forward(self, batch):
        # batch, time, freq, channels
        x = batch.permute(0, 3, 1, 2)
        x = self.bn0(x)
        x = F.relu(self.conv1(x))
        # (batch_size=32, channels=1, seq_len, input_dim=216)
        x = self.bn1(x).squeeze(1)
        return x


class ConformerCell(nn.Module):
    def __init__(self, input_dim=216, hidden_dim=128, output_dim=128):
        super().__init__()

        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=8,
            ffn_dim=hidden_dim,
            num_layers=12,
            depthwise_conv_kernel_size=31,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, batch, mask):
        lengths = mask.sum(dim=-1)
        x = self.conformer(batch, lengths=lengths)
        x = self.norm(x[0])
        x = self.output_layer(x)
        return x


class ConformerSequenceEncoder(BaseSequence):
    def __init__(self, input_dim=216, hidden_dim=128, num_layers=1, nhead=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.conformer_cell = ConformerCell(input_dim, 128, hidden_dim)
        # self.position_encoding = nn.Parameter(
        #    torch.zeros(1, 2000, hidden_dim)
        self.position_encoding = self.create_sinusoidal_positional_encoding(
            3000, hidden_dim
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                batch_first=True,
                dim_feedforward=hidden_dim,
            ),
            num_layers=num_layers,
        )

    def forward(self, sequence_batch, sequence_mask=None):
        padded_sequences = sequence_batch
        x = self.conformer_cell(padded_sequences, sequence_mask)
        x = self.add_positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=sequence_mask)
        return x


class HCQTConformer(nn.Module):
    def __init__(
        self,
        input_dim=216,
        hidden_dim=128,
        output_dim=12,
    ):
        super().__init__()

        # Convolutional part of encoder
        self.bn0 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size=(5, 5), padding="same", dtype=torch.float
        )
        self.bn1 = nn.BatchNorm2d(1)

        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=8,
            ffn_dim=hidden_dim,
            num_layers=12,
            depthwise_conv_kernel_size=31,
        )
        self.norm = nn.LayerNorm(input_dim)
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        x = batch.permute(0, 3, 1, 2)
        x = self.bn0(x)
        x = F.relu(self.conv1(x))
        # (batch_size=32, channels=1, seq_len=128, input_dim=216)
        x = self.bn1(x).squeeze(1)
        x = self.conformer(
            x, lengths=torch.tensor([128 for _ in range(x.shape[0])], device=x.device)
        )
        x = self.norm(x[0])
        x = self.output_layer(x)
        return x


class ChromaPredictor(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate_accuracy(self, outputs, labels):
        binarized_outputs = (outputs > 0.5).float()
        return (binarized_outputs == labels[:, :, :12]).float().mean()

    def generic_step(self, batch):
        outputs = F.sigmoid(self(batch[0]))
        loss = F.binary_cross_entropy(outputs, batch[1][:, :, :12], reduction="mean")
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, outputs = self.generic_step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self.generic_step(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        accuracy = self.calculate_accuracy(outputs, batch[1])
        self.log(
            "val_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs = self.generic_step(batch)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        accuracy = self.calculate_accuracy(outputs, batch[1])
        self.log(
            "test_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.model(x)


class ChromaInference:
    def __init__(self, model, patch_size=128, hop_size=64):
        self.model = model
        self.patch_size = patch_size
        self.hop_size = hop_size

    def _calculate_predictions(self, patches):
        with torch.no_grad():
            predictions = []
            for i in range(len(patches)):
                pred = self.model(patches[i].unsqueeze(0))
                predictions.append(pred.squeeze(0))
            return predictions

    def _convert_patches_to_one_tensor(self, patches, track_length=None):
        # Get model predictions for all patches
        predictions = self._calculate_predictions(patches)

        # If track_length is not provided, estimate it from patches and hop_size
        if track_length is None:
            # Calculate approximate track length based on number of patches and hop size
            track_length = (len(predictions) - 1) * self.hop_size + self.patch_size

        # Initialize output tensor and weights tensor
        output_dim = predictions[0].shape[
            -1
        ]  # Get the output dimension (e.g., 12 for chroma)
        output_tensor = torch.zeros(
            (track_length, output_dim), device=predictions[0].device
        )
        weights = torch.zeros(track_length, device=predictions[0].device)

        # Add each patch prediction to the output tensor with appropriate weights
        for i, pred in enumerate(predictions):
            # Calculate the start and end positions for this patch
            start_pos = i * self.hop_size
            end_pos = min(start_pos + self.patch_size, track_length)

            # Add prediction values to output tensor
            output_tensor[start_pos:end_pos] += pred[: end_pos - start_pos]
            # Add weights (1.0 for each position that receives a prediction)
            weights[start_pos:end_pos] += 1.0

        # Divide by weights to get weighted average at each position
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        output_tensor = output_tensor / (weights.unsqueeze(-1) + eps)

        return output_tensor

    def __call__(self, audio):
        patches = AudioPatches(audio, self.patch_size, self.hop_size)
        # Get the original track length
        track_length = audio.features.shape[0]
        # Pass the patches and track length to our conversion function
        output = self._convert_patches_to_one_tensor(patches, track_length)
        if output.shape[0] > track_length:
            output = output[:track_length]
        return output


def main():
    parser = argparse.ArgumentParser(description='Run chroma inference on audio files')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Path to save output features (optional)')
    parser.add_argument('--patch-size', type=int, default=128, help='Size of patches for inference')
    parser.add_argument('--hop-size', type=int, default=64, help='Hop size between patches')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension of the model')
    parser.add_argument('--output-dim', type=int, default=12, help='Output dimension (chroma bins)')
    
    args = parser.parse_args()
    
    # Load model
    model = ChromaPredictor(
        HCQTConformer(
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
        ).float()
    )
    model.load_state_dict(torch.load(args.model))
    
    # Create inference object
    inference = ChromaInference(model, patch_size=args.patch_size, hop_size=args.hop_size)
    
    # Load and process audio
    track = AudioTrack(audio_path=args.audio)
    features = inference(track)
    
    # Save features if output path is provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, features.numpy())
        print(f"Features saved to {output_path}")
    else:
        print("Features shape:", features.shape)
        print("Features:", features)

if __name__ == '__main__':
    main()
