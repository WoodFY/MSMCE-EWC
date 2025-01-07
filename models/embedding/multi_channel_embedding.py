import torch
import torch.nn as nn


class MultiChannelEmbedding(nn.Module):

    def __init__(self, spectrum_dim, embedding_channels, embedding_dim):
        super().__init__()

        self.spectrum_dim = spectrum_dim
        self.embedding_channels = embedding_channels
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(spectrum_dim, 2048),
            # nn.BatchNorm1d(2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, embedding_dim),
        )

        self.channel_embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=embedding_channels // 2, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embedding_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=embedding_channels // 2, out_channels=embedding_channels, kernel_size=3, padding='same'),
        )

    def forward(self, x):
        # encoded: (batch_size, embedding_dim)
        encoded = self.encoder(x)
        # encoded: (batch_size, 1, embedding_dim)
        encoded = encoded.unsqueeze(1)

        embedded = self.channel_embedding(encoded)

        out = torch.cat([encoded, embedded], dim=1)

        return out