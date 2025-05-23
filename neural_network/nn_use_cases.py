# MLE Interview Preparation Codebase

## Time Series Forecasting (Neural Network)

# nn_time_series.py
import torch
import torch.nn as nn
import torch.optim as optim

class TimeSeriesNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimeSeriesNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

## Anomaly Detection (Autoencoder-based)

# anomaly_detection_nn.py
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(AnomalyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

## Computer Vision (Edge, Texture, Shape Detection)

# edge_detection_cnn.py
import torchvision.transforms as transforms
import torchvision.models as models
class EdgeDetectionCNN(nn.Module):
    def __init__(self):
        super(EdgeDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

## NLP (Speech Recognition / Complex Word Relationships)

# speech_recognition_rnn.py
import torchaudio
class SpeechRecognitionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpeechRecognitionRNN, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out[:, -1, :])
