import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchtext
from torchtext.data.utils import get_tokenizer

import pandas as pd

import sys

class PositionalEncoding(nn.Module):
    def __init__(self, embedded_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encoder = torch.zeros(max_len, embedded_size)
        position = torch.arrange(0, max_len, dtype=torch.float).unsqueeze(1)
        normalizer = torch.exp(torch.arange(0, embedded_size, 2).float() * (-math.log(10000.0) / embedded_size))

        positional_encoder[:, 0::2] = torch.sin(position * normalizer)
        positional_encoder[:, 1::2] = torch.cos(position * normalizer)

        positional_encoder = positional_encoder.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_encoder', positional_encoder)

    def forward(selfself, x):
        x_pos = x + self.positional_encoder[:x.size(0), :]
        return self.dropout(x_pos)


class Transformer(nn.Module):

    def __init__(self, n_words, embedded_size, n_heads, n_hidden, n_layers, dropout=0.5, device='cuda'):
        super(Transformer, self).__init__()

        self.device = device
        self.model_type = 'Transformer'
        self.src_mask = None
        self.positional_encoder = PositionalEncoding(embedded_size, dropout)

        encoder_layers = nn.TransformerEncoderLayer(embedded_size, n_heads, n_hidden, dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_words, embedded_size)
        self.embedded_size = embedded_size

        self.decoder = nn.Linear(embedded_size, n_words)

        self.init_weights()


    def _generate_square_mask_(self, size):
        mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform(-init_range, init_range)


    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_mask_(len(src)).to(self.device)
            self.src_mask = mask

        src_embeddings = self.encoder(src) * math.sqrt(self.embedded_size)
        src_positional_embeddings = self.positional_encoder(src_embeddings)
        output_embeddings = self.transformer_encoder(src_positional_embeddings)
        output = self.decoder(output_embeddings)
        return output



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print('device is:', device)

    dataset_path = 'data/dataset.csv'

    dataset = pd.read_csv(dataset_path)



