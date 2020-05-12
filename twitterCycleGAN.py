import math
import torch
import torch.nn as nn
import torch.nn.functional as F

###################### USED IN DATA PREPARATION ######################
import torchtext
from torchtext.data.utils import get_tokenizer
import re
######################################################################

import time
import pandas as pd
import numpy as np

import json
import os
import csv
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


class Generator(nn.Module):

    def __init__(self, input_vocab_size, output_vocab_size, embedded_size, n_heads, n_hidden, n_layers,
                 dropout=0.5, device=torch.device('cuda'), max_len=50, pad=0, sos=1, eos=2):
        super(Generator, self).__init__()

        self.device = device
        self.model_type = 'Transformer'
        self.sos = sos
        self.eos = eos
        self.max_len = max_len

        self.embedded_size = embedded_size
        self.input_vocab_size = input_vocab_size
        self.src_encoder = nn.Embedding(input_vocab_size, embedded_size, padding_idx=pad)

        self.positional_encoder = PositionalEncoding(embedded_size, dropout)

        encoder_layers = nn.TransformerEncoderLayer(embedded_size, n_heads, n_hidden, dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        decoder_layers = nn.TransformerDecoderLayer(embedded_size, n_heads, n_hidden, dropout)

        self.trg_encoder = nn.Embedding(output_vocab_size, embedded_size, padding_idx=pad)

        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, n_layers)

        self.trg_decoder = nn.Linear(embedded_size, output_vocab_size)

        self.init_weights()


    def init_weights(self):
        init_range = 0.1
        self.src_encoder.weight.data.uniform_(-init_range, init_range)
        self.trg_encoder.weight.data.uniform_(-init_range, init_range)
        self.trg_decoder.bias.data.zero_()
        self.trg_decoder.weight.data.uniform_(-init_range, init_range)


    def _generate_padding_mask_(self, batch_size, max_len, lens):
        mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        for idx, length in enumerate(lens):
            mask[idx][:length] = False
        return mask


    def forward_one_word(self, src, trg, src_lens, trg_lens):
        src_embeddings = self.src_encoder(src) * math.sqrt(self.embedded_size)
        src_positional_embeddings = self.positional_encoder(src_embeddings)

        max_len, batch_size = list(src.size())
        src_padding_mask = self._generate_padding_mask_(batch_size, max_len, src_lens).to(self.device)
        memory = self.transformer_encoder(src_positional_embeddings, src_key_padding_mask=src_padding_mask)

        trg_embeddings = self.trg_encoder(trg) * math.sqrt(self.embedded_size)
        trg_positional_embeddings = self.positional_encoder(trg_embeddings)

        trg_padding_mask = self._generate_padding_mask_(batch_size, max_len, trg_lens).to(self.device)

        output_embeddings = self.transformer_decoder(trg_positional_embeddings, memory,
                                                     memory_key_padding_mask=src_padding_mask,
                                                     tgt_key_padding_mask=trg_padding_mask,)
        output = self.trg_decoder(output_embeddings)

        return output

    def forward(self, src, src_lens):
        max_len, batch_size = list(src.size())

        trg = torch.zeros(max_len, batch_size, dtype=torch.long).to(self.device)
        trg[0,:] = self.sos

        trg_lens = torch.ones(batch_size, dtype=torch.long).to(self.device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)

        for idx in range(1, max_len-1):
            output = self.forward_one_word(src, trg, src_lens, trg_lens)
            pred = output[idx, :, :].max(1).indices
            finished_now = (pred == self.eos)
            trg_lens[finished_now] = idx+1
            is_finished = is_finished | finished_now
            trg[idx][~is_finished] = pred[~is_finished]

        trg[max_len-1][~is_finished] = self.eos
        trg_lens[~is_finished] = max_len

        return trg, trg_lens


class Discriminator(nn.Module):

    def __init__(self, vocab_size, embedded_size=200, n_heads=2, n_hidden=200, n_layers=2,
                 dropout=0.2, max_len=50, device=torch.device('cuda'), pad=0):
        super(Discriminator, self).__init__()
        self.device = device
        self.embedded_size = embedded_size
        self.vocab_size = vocab_size

        self.src_encoder = nn.Embedding(vocab_size, embedded_size, padding_idx=pad)
        self.positional_encoder = PositionalEncoding(embedded_size, dropout)

        encoder_layers = nn.TransformerEncoderLayer(embedded_size, n_heads, n_hidden, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.decoder = nn.Linear(embedded_size*max_len, 2)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.src_encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def _generate_padding_mask_(self, batch_size, max_len, lens):
        mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        for idx, length in enumerate(lens):
            mask[idx][:length] = False
        return mask

    def forward(self, src, src_lens):
        src_embeddings = self.src_encoder(src) * math.sqrt(self.embedded_size)
        src_positional_embeddings = self.positional_encoder(src_embeddings)

        max_len, batch_size = list(src.size())
        src_padding_mask = self._generate_padding_mask_(batch_size, max_len, src_lens).to(self.device)
        src_embedding = self.transformer_encoder(src_positional_embeddings,
                                                 src_key_padding_mask=src_padding_mask
                                                 ).transpose(0,1).view(batch_size, -1).contiguous()
        outputs = self.decoder(src_embedding)
        return outputs

def tokens_to_sentences(src, src_lens, vocab_itos):
    sentences_tokens = src.transpose(0, 1).tolist()
    sentences = []
    for idx, sentence_tokens in enumerate(sentences_tokens):
        sentence = list(map(lambda x: vocab_itos[x], sentence_tokens[:src_lens[idx]]))
        sentences.append(sentence)
    return sentences

def train_model(model_name, user1, user2, device):

    ################### prepare the data ###################

    def prepare_data(user, batch_size, max_len, device):
        train_df = pd.read_csv('data/' + user + '/train.csv')
        val_df = pd.read_csv('data/' + user + '/val.csv')
        test_df = pd.read_csv('data/' + user + '/test.csv')

        vocab_itos = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        vocab_stoi = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}

        train_tokens = torch.zeros(len(train_df), max_len, dtype=torch.long)
        val_tokens = torch.zeros(len(val_df), max_len, dtype=torch.long)
        test_tokens = torch.zeros(len(test_df), max_len, dtype=torch.long)

        train_lens = torch.zeros(len(train_df), dtype=torch.long)
        val_lens = torch.zeros(len(val_df), dtype=torch.long)
        test_lens = torch.zeros(len(test_df), dtype=torch.long)

        for i, tweet in enumerate(train_df['text']):
            words = tweet.split(' ')
            if len(words) > max_len - 2:
                words = words[:max_len - 2]
            words = ['<sos>'] + words + ['<eos>']
            train_lens[i] = len(words)
            for j, word in enumerate(words):
                if word not in vocab_stoi:
                    token = len(vocab_stoi)
                    vocab_stoi[word] = token
                    vocab_itos[token] = word
                else:
                    token = vocab_stoi[word]
                train_tokens[i][j] = token

        for i, tweet in enumerate(val_df['text']):
            words = tweet.split(' ')
            if len(words) > max_len - 2:
                words = words[:max_len - 2]
            words = ['<sos>'] + words + ['<eos>']
            val_lens[i] = len(words)
            for j, word in enumerate(words):
                if word not in vocab_stoi:
                    token = vocab_stoi['<unk>']
                else:
                    token = vocab_stoi[word]
                val_tokens[i][j] = token

            for i, tweet in enumerate(test_df['text']):
                words = tweet.split(' ')
                if len(words) > max_len - 2:
                    words = words[:max_len - 2]
                words = ['<sos>'] + words + ['<eos>']
                test_lens[i] = len(words)
                for j, word in enumerate(words):
                    if word not in vocab_stoi:
                        token = vocab_stoi['<unk>']
                    else:
                        token = vocab_stoi[word]
                    val_tokens[i][j] = token

        def batchify(tokens, lens, batch_size):
            n_batches = tokens.size(0) // batch_size
            data = tokens.narrow(0, 0, n_batches * batch_size)
            lens = lens.narrow(0, 0, n_batches*batch_size)
            data = data.view(n_batches, batch_size, -1).transpose(1,2).contiguous()
            lens = lens.view(n_batches, batch_size).contiguous()
            return data.to(device), lens.to(device)

        train_data, train_lens = batchify(train_tokens, train_lens, batch_size)
        val_data, val_lens = batchify(val_tokens, val_lens, batch_size)
        test_data, test_lens = batchify(test_tokens, test_lens, batch_size)

        return train_data, val_data, test_data, train_lens, val_lens, test_lens, vocab_itos
    ################### training settings ###################

    batch_size = 20
    max_len = 50

    (user1_train_data, user1_val_data, user1_test_data,
     user1_train_lens, user1_val_lens, user1_test_lens, user1_vocab_itos) = prepare_data(user1, batch_size=batch_size,
                                                                                       max_len=max_len, device=device)

    (user2_train_data, user2_val_data, user2_test_data,
     user2_train_lens, user2_val_lens, user2_test_lens, user2_vocab_itos) = prepare_data(user2, batch_size=batch_size,
                                                                                       max_len=max_len, device=device)

    print(user1_train_data.size())

    user1_vocab_size = len(user1_vocab_itos)
    user2_vocab_size = len(user2_vocab_itos)

    embedded_size = 200
    n_heads = 2
    n_hidden = 200
    n_layers = 2
    dropout = 0.2

    model = Generator(user1_vocab_size, user2_vocab_size, embedded_size, n_heads,
                        n_hidden, n_layers, dropout, device=device).to(device)

    example = user1_train_data[0,:,:1]
    example_len = user1_train_lens[0,:1]
    output_example, output_len = model(example, user1_train_lens[0,:1])

    print(tokens_to_sentences(example, example_len, user1_vocab_itos))

    print(output_example.size())
    output_sentence = tokens_to_sentences(output_example, output_len, user2_vocab_itos)
    print(output_sentence)

    discriminator = Discriminator(user2_vocab_size, device=device).to(device)

    example_bernie = user2_train_data[0,:,:1]

    print('True Bernie')
    print(F.softmax(discriminator(example_bernie, user2_train_lens[0,:1]), dim=1))

    print('Translated to Bernie')
    print(F.softmax(discriminator(output_example, output_len), dim=1))

    return None, None
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

    ################ saving the model ################

    model_params = model.state_dict()

    model_kwargs = {'input_vocab_size': user1_vocab_size, 'output_vocab_size': user2_vocab_size,
                    'embedded_size': embedded_size, 'n_heads': n_heads, 'n_hidden': n_hidden, 'n_layers': n_layers,
                    'dropout': dropout, 'device': device}

    save_model(model_name, model_params, model_kwargs)

    return best_model, None

def save_model(model_name, model_params, model_kwargs):

    params_path = 'models/' + model_name + '_params.pt'

    torch.save(model_params, params_path)

    kwargs_path = 'models/' + model_name + '_kwargs.json'

    with open(kwargs_path, 'w') as kwargs_file:
        json.dump(model_kwargs, kwargs_file, sort_keys=True, indent=4)

    print(model_name, 'saved')


def load_model(model_name, device):

    kwargs_path = 'models/' + model_name + '_kwargs.json'
    with open(kwargs_path, 'r') as kwargs_file:
        model_kwargs = json.load(kwargs_file)

    model = Generator(**model_kwargs).to(device)

    params_path = 'models/' + model_name + '_params.pt'

    model.load_state_dict(torch.load(params_path))
    model.eval()

    print(model_name, 'loaded with device', str(model.device))

    return model


def test_model(model_name, device):

    model = load_model(model_name, device)

    print(model.state_dict())


def prepare_text(s):
    words = s.split(' ')
    websites = []
    for word in words:
        if 'http' in word:
            websites.append(word)
    for website in websites:
        words.remove(website)
    s = " ".join(words)
    words = re.split(r'(\W)', s)
    n_spaces, n_blanks, n_breaks, n_r = 0, 0, 0, 0
    for word in words:
        if word == ' ':
            n_spaces += 1
        if word == '':
            n_blanks += 1
        if word == '\n':
            n_breaks += 1
        if word == '\r':
            n_r += 1
    for space in range(n_spaces):
        words.remove(" ")
    for blank in range(n_blanks):
        words.remove('')
    for line_break in range(n_breaks):
        words.remove('\n')
    for r in range(n_r):
        words.remove('\r')
    s = " ".join(words)
    return s

def prepare_user_data(user, n_tweets):

    dataset = pd.read_csv('data/dataset.csv', lineterminator='\n')

    data = dataset[dataset['user'] == user]['text']

    data = pd.Series(map(prepare_text, data))
    data = pd.DataFrame(data, columns=['text'])

    data['text'].replace('', np.nan, inplace=True)
    data.dropna(subset=['text'], inplace=True)

    data = data[:n_tweets]

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    pct_train, pct_val = (0.8, 0.1)
    n_train = math.floor(n_tweets * pct_train)
    n_val = math.floor(n_tweets * pct_val)

    data_train = data[:n_train].reset_index()
    data_val = data[n_train:n_train + n_val].reset_index()
    data_test = data[n_train + n_val:].reset_index()

    os.mkdir('data/' + user)

    data_train.to_csv('data/' + user + '/train.csv', index_label='index')
    data_val.to_csv('data/' + user + '/val.csv', index_label='index')
    data_test.to_csv('data/' + user + '/test.csv', index_label='index')


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print('device is:', device)

    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    #random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    mode = sys.argv[1]

    if mode == 'train':
        model_name, user1, user2 = sys.argv[2], sys.argv[3], sys.argv[4]
        model1, model2 = train_model(model_name=model_name, user1=user1, user2=user2, device=device)

    if mode == 'test':
        model_name = sys.argv[2]
        test_model(model_name=model_name, device=device)

    if mode == 'prepare_user_data':
        user = sys.argv[2]
        n_tweets = int(sys.argv[3])
        prepare_user_data(user, n_tweets)


