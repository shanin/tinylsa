
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.bn0 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size=(5, 5), padding="same", dtype=torch.float
        )
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(
            1, 4, kernel_size=(3, 3), padding="same", dtype=torch.float
        )
        self.bn2 = nn.BatchNorm2d(4)

    def forward(self, x):
        x = self.bn0(x)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        return x


class HarmonicEncoder(nn.Module):
    def __init__(self):
        super(HarmonicEncoder, self).__init__()

        self.chord_conv1 = nn.Conv2d(
            4, 1, kernel_size=(1, 1), padding="same", dtype=torch.float
        )
        self.chord_bn1 = nn.BatchNorm2d(1)
        self.chord_conv2 = nn.Conv2d(
            1, 72, kernel_size=(1, 216), padding="valid", dtype=torch.float
        )
        self.chord_bn2 = nn.BatchNorm2d(72)

        # Recurrent part of harmonic encoder
        self.chord_rnn1 = nn.GRU(72, 128, bidirectional=True, batch_first=True)
        self.chord_r1bn = nn.BatchNorm2d(1)
        self.chord_rnn2 = nn.GRU(256, 128, bidirectional=True, batch_first=True)
        self.chord_r2bn = nn.BatchNorm2d(1)

    def forward(self, x):

        x = self.chord_bn1(F.relu(self.chord_conv1(x)))
        x = self.chord_bn2(F.relu(self.chord_conv2(x)))

        x = x.squeeze()
        x = x.transpose(1, 2)

        x = self.chord_rnn1(x)[0]
        x = x.reshape(-1, 1, x.size(1), x.size(2))
        x = self.chord_r1bn(x)
        x = x.squeeze()

        x = self.chord_rnn2(x)[0]
        x = x.reshape(-1, 1, x.size(1), x.size(2))
        x = self.chord_r2bn(x)
        x = x.squeeze()

        return x


class BeatBranch(nn.Module):
    def __init__(self):
        super(BeatBranch, self).__init__()

        self.beat_conv1 = nn.Conv2d(
            4, 4, kernel_size=(3, 3), padding="same", dtype=torch.float
        )
        self.beat_bn1 = nn.BatchNorm2d(4)
        self.beat_conv2 = nn.Conv2d(
            4, 1, kernel_size=(3, 3), padding="same", dtype=torch.float
        )
        self.beat_bn2 = nn.BatchNorm2d(1)
        self.beat_maxpool = nn.MaxPool2d(kernel_size=(1, 2))
        self.beat_rnn1 = nn.GRU(108, 64, bidirectional=True, batch_first=True)
        self.beat_r1bn = nn.BatchNorm2d(1)
        self.beat_rnn2 = nn.GRU(128, 64, bidirectional=True, batch_first=True)
        self.beat_r2bn = nn.BatchNorm2d(1)
        self.beat_predictor = nn.Linear(128, 2)

    def forward(self, x):
        x = self.beat_bn1(F.relu(self.beat_conv1(x)))
        x = self.beat_bn2(F.relu(self.beat_conv2(x)))
        x = self.beat_maxpool(x).squeeze()
        x = self.beat_rnn1(x)[0]
        x = x.reshape(-1, 1, x.size(1), x.size(2))
        x = self.beat_r1bn(x)
        x = x.squeeze()
        x = self.beat_rnn2(x)[0]
        x = x.reshape(-1, 1, x.size(1), x.size(2))
        x = self.beat_r2bn(x)
        x = F.softmax(self.beat_predictor(x), dim=-1)
        return x

class BeatCrema(nn.Module):
    # inspired by Duran, de la Cuadra (2020)
    def _configure(self):
        self.n_classes = 170
        self.segment_size = 100

    def __init__(self):
        super(BeatCrema, self).__init__()
        self._configure()

        self.shared = SharedEncoder()
        self.beat_branch = BeatBranch()
        self.harmonic_encoder = HarmonicEncoder()

        # structured learning
        self.pitch_predictor = nn.Linear(256, 12)
        self.root_predictor = nn.Linear(256, 13)
        self.bass_predictor = nn.Linear(256, 13)

        # RNN chord prediction
        self.chord_r1bn = nn.BatchNorm2d(1)
        self.chord_rnn1 = nn.GRU(296, 128, bidirectional=True, batch_first=True)
        self.chord_r2bn = nn.BatchNorm2d(1)
        self.chord_rnn2 = nn.GRU(256, 128, bidirectional=True, batch_first=True)

        # self.codecbn = nn.BatchNorm2d(1)
        self.chord_predictor = nn.Linear(256, self.n_classes)

    def forward(self, x):
        x = x.transpose(1, 3).transpose(2, 3)

        # encoder
        x = self.shared(x)
        beat_predictions = self.beat_branch(x)
        x = self.harmonic_encoder(x)

        # structure learning
        pitch_classes = torch.sigmoid(self.pitch_predictor(x))
        root_class = F.softmax(self.root_predictor(x), dim=-1)
        bass_class = F.softmax(self.bass_predictor(x), dim=-1)

        detached_beat = beat_predictions.squeeze().detach()
        codec = torch.cat(
            [x, pitch_classes, root_class, bass_class, detached_beat],
            dim=-1,
        )

        x = self.chord_rnn1(codec)[0]
        x = x.reshape(-1, 1, x.size(1), x.size(2))
        x = self.chord_r1bn(x)
        x = x.squeeze()

        x = self.chord_rnn2(x)[0]
        x = x.reshape(-1, 1, x.size(1), x.size(2))
        x = self.chord_r2bn(x)
        codec = x.squeeze()

        codec = codec.reshape(-1, 1, codec.size(1), codec.size(2))
        chord_prediction = F.softmax(self.chord_predictor(codec), dim=-1)

        """
        return {
            "chord": chord_prediction,
            "pitch": pitch_classes,
            "root": root_class,
            "bass": bass_class,
            "beats": beat_predictions,
            "chord_embedding": codec,
        }
        """
        return pitch_classes



model_config = {
    'batch_size': 64,
    'segment_length': 100,
    'lr': 0.001,
    'label_smoothing': 0.2,
    'weight_decay': 0.01
}