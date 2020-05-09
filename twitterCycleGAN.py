import math
import torch
import torch.nn as nn
import torch.nn.functional as F

###################### USED IN DATA PREPARATION ######################
import torchtext
from torchtext.data.utils import get_tokenizer
######################################################################

import time
import pandas as pd

import sys

class PositionalEncoding(nn.Module):
    def __init__(self, embedded_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encoder = torch.zeros(max_len, embedded_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        normalizer = torch.exp(torch.arange(0, embedded_size, 2).float() * (-math.log(10000.0) / embedded_size))

        positional_encoder[:, 0::2] = torch.sin(position * normalizer)
        positional_encoder[:, 1::2] = torch.cos(position * normalizer)

        positional_encoder = positional_encoder.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_encoder', positional_encoder)

    def forward(self, x):
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
        self.n_words = n_words

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
        self.decoder.weight.data.uniform_(-init_range, init_range)


    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_mask_(len(src)).to(device)
            self.src_mask = mask

        src_embeddings = self.encoder(src) * math.sqrt(self.embedded_size)
        src_positional_embeddings = self.positional_encoder(src_embeddings)
        output_embeddings = self.transformer_encoder(src_positional_embeddings)
        output = self.decoder(output_embeddings)
        return output


def train(device):

    ################### prepare the data ###################

    def prepare_data(batch_size, eval_batch_size, device='cuda'):
        TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                    init_token='<sos>', eos_token='<eos>', lower=True)
        train_text, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
        TEXT.build_vocab(train_text)

        def batchify(data, batch_size):
            data = TEXT.numericalize([data.examples[0].text])
            n_batches = data.size(0) // batch_size
            data = data.narrow(0, 0, n_batches * batch_size)
            data = data.view(batch_size, -1).t().contiguous()
            return data.to(device)

        train_data = batchify(train_text, batch_size)
        val_data = batchify(val_txt, eval_batch_size)
        test_data = batchify(test_txt, eval_batch_size)

        n_words = len(TEXT.vocab.stoi)

        return train_data, val_data, test_data, n_words

    bptt = 35
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    ################### training settings ###################

    train_data, val_data, test_data, n_words = prepare_data(batch_size=20, eval_batch_size=10, device=device)

    embedded_size = 200
    n_heads = 2
    n_hidden = 200
    n_layers = 2
    dropout = 0.2

    model = Transformer(n_words, embedded_size, n_heads, n_hidden, n_layers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def train_epoch(train_model, data_source):
        train_model.train()
        total_loss = 0
        start_time = time.time()

        for batch, i in enumerate(range(0, data_source.size(0) -1, bptt)):
            data, targets = get_batch(data_source, i)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, n_words), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                time_elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| lr {:02.2f} | ms/batch {:5.2f} '
                      '| loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, len(data_source) // bptt, scheduler.get_lr()[0],
                                                           time_elapsed * 1000 / log_interval, cur_loss,
                                                           math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(eval_model, data_source):
        eval_model.eval()
        total_loss = 0
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                output = eval_model(data)
                output_flat = output.view(-1, n_words)
                total_loss += len(data) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    best_val_loss = float('inf')
    epochs = 3
    best_model = None

    ################### training loop ###################

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_epoch(model, train_data)
        val_loss = evaluate(model, val_data)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s '
              '| valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss,
                                                                math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    ################### evaluation ###################

    test_loss = evaluate(best_model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
    print('=' * 89)

    return best_model


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print('device is:', device)

    #dataset_path = 'data/dataset.csv'
    #dataset = pd.read_csv(dataset_path)

    mode = sys.argv[1]

    if mode == 'train':
        model = train(device=device)


