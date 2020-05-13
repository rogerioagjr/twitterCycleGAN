import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

    def __init__(self, vocab_size, embedded_size=32, n_heads=1, n_hidden=64, n_layers=1,
                 dropout=0.1, max_len=50, device=torch.device('cuda'), pad=0):
        super(Discriminator, self).__init__()
        self.device = device
        self.embedded_size = embedded_size
        self.vocab_size = vocab_size

        self.src_encoder = nn.Embedding(vocab_size, embedded_size, padding_idx=pad)
        self.positional_encoder = PositionalEncoding(embedded_size, dropout)

        encoder_layers = nn.TransformerEncoderLayer(embedded_size, n_heads, n_hidden, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.decoder = nn.Linear(embedded_size*max_len, 1)

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
                                                 ).transpose(0,1).reshape(batch_size, -1).contiguous()
        outputs = self.decoder(src_embedding)
        return torch.sigmoid(outputs)

def tokens_to_sentences(src, src_lens, vocab_itos):
    sentences_tokens = src.transpose(0, 1).tolist()
    sentences = []
    for idx, sentence_tokens in enumerate(sentences_tokens):
        sentence = list(map(lambda x: vocab_itos[x], sentence_tokens[:src_lens[idx]]))
        sentences.append(sentence)
    return sentences

def train_model(model_name, user1, user2, n_epochs, device):

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

    user1_vocab_size = len(user1_vocab_itos)
    user2_vocab_size = len(user2_vocab_itos)

    embedded_size = 512
    n_heads = 8
    n_hidden = 2048
    n_layers = 6
    dropout = 0.1

    generator = Generator(user1_vocab_size, user2_vocab_size, embedded_size, n_heads,
                        n_hidden, n_layers, dropout, device=device).to(device)

    discriminator = Discriminator(user2_vocab_size, device=device).to(device)

    ################# training loop #################

    lr = 0.0002
    beta1 = 0.5

    tweets_list = []
    G_losses = []
    D_losses = []

    iters = 0

    real_label = 1
    fake_label = 0

    criterion = nn.BCELoss()

    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    n_batches = user1_train_data.size(0)

    n_fixed = 5
    fixed_src = user1_train_data[0,:,:n_fixed]
    fixed_lens = user1_train_lens[0,:n_fixed]

    print("Starting Training Loop...")
    for epoch in range(n_epochs):
        for batch_idx in range(n_batches):
            ##############################################################
            #                   Update Discriminator                     #
            ##############################################################
            ## Train on real Sanders tweets

            user1_src = user1_train_data[batch_idx]
            user1_lens = user1_train_lens[batch_idx]

            user2_src = user2_train_data[batch_idx]
            user2_lens = user2_train_lens[batch_idx]

            discriminator.zero_grad()

            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(user2_src, user2_lens).view(-1)

            err_discriminator_real = criterion(output, label)
            err_discriminator_real.backward()
            D_y = output.mean().item()

            ## Train on fake Sanders tweets

            fake_src, fake_lens = generator(user1_src, user1_lens)
            label.fill_(fake_label)
            output = discriminator(fake_src.detach(), fake_lens.detach()).view(-1)
            err_discriminator_fake = criterion(output, label)
            err_discriminator_fake.backward()
            D_G_x1 = output.mean().item()
            err_discriminator = err_discriminator_real + err_discriminator_fake
            optimizerD.step()

            ##############################################################
            #                     Update Generator                       #
            ##############################################################

            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake_src, fake_lens).view(-1)
            err_generator = criterion(output, label)
            err_generator.backward()
            D_G_x2 = output.mean().item()
            optimizerG.step()

            ##############################################################
            #                   Output Training Stats                    #
            ##############################################################

            if batch_idx % 20 == 0 or batch_idx == n_batches-1:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(y): '
                      '%.4f\tD(G(x)): %.4f / %.4f' % (epoch, n_epochs, batch_idx, n_batches, err_discriminator.item(),
                                                      err_generator.item(), D_y, D_G_x1, D_G_x2))

            # Save losses for plotting
            G_losses.append(err_generator.item())
            D_losses.append(err_discriminator.item())

            iters += 1

        with torch.no_grad():
            fake_src, fake_lens = generator(fixed_src, fixed_lens)
        tweets_list.append(tokens_to_sentences(fake_src, fake_lens, user2_vocab_itos))

    ########### show translation evolution ###########

    print('=' * 100)
    print('Source tweets examples')
    print('=' * 100)

    fixed_sentences = tokens_to_sentences(fixed_src, fixed_lens, user1_vocab_itos)
    for sentence in fixed_sentences:
        print(" ".join(sentence))
    print('='*100)

    print('=' * 100)
    print('Translated tweets examples')
    print('=' * 100)

    for epoch, sentences in enumerate(tweets_list):
        print('epoch:', epoch)
        for sentence in sentences:
            print(" ".join(sentence))
        print('='*100)

    ################## plot results ##################

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('images/' + model_name + '_results.eps', format='eps', dpi=1200)

    ################ saving the model ################

    generator_params = generator.state_dict()

    generator_kwargs = {'input_vocab_size': user1_vocab_size, 'output_vocab_size': user2_vocab_size,
                    'embedded_size': embedded_size, 'n_heads': n_heads, 'n_hidden': n_hidden, 'n_layers': n_layers,
                    'dropout': dropout}

    save_model(model_name + '_generator', generator_params, generator_kwargs)

    discriminator_params = discriminator.state_dict()

    discriminator_kwargs = {'vocab_size':user2_vocab_size}

    save_model(model_name + '_discriminator', discriminator_params, discriminator_kwargs)


    return discriminator, generator

def save_model(model_name, model_params, model_kwargs):

    params_path = 'models/' + model_name + '_params.pt'

    torch.save(model_params, params_path)

    kwargs_path = 'models/' + model_name + '_kwargs.json'

    with open(kwargs_path, 'w') as kwargs_file:
        json.dump(model_kwargs, kwargs_file, sort_keys=True, indent=4)

    print(model_name, 'saved')


def load_model(model_name, device):

    generator_kwargs_path = 'models/' + model_name + '_generator_kwargs.json'
    with open(generator_kwargs_path, 'r') as generator_kwargs_file:
        generator_kwargs = json.load(generator_kwargs_file)

    generator = Generator(**generator_kwargs).to(device)

    generator_params_path = 'models/' + model_name + '_generator_params.pt'

    generator.load_state_dict(torch.load(generator_params_path))
    generator.eval()

    discriminator_kwargs_path = 'models/' + model_name + '_discriminator_kwargs.json'
    with open(discriminator_kwargs_path, 'r') as discriminator_kwargs_file:
        discriminator_kwargs = json.load(discriminator_kwargs_file)

    discriminator = Discriminator(**discriminator_kwargs).to(device)

    discriminator_params_path = 'models/' + model_name + '_discriminator_params.pt'

    discriminator.load_state_dict(torch.load(discriminator_params_path))
    discriminator.eval()

    print(model_name, 'loaded with device', str(generator.device))

    return generator, discriminator


def test_model(model_name, device):

    generator, discriminator = load_model(model_name, device)

    print(generator.state_dict())
    print(discriminator.state_dict())


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
        model_name, user1, user2, n_epochs = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
        discriminator, generator = train_model(model_name=model_name, user1=user1, user2=user2,
                                               n_epochs=n_epochs, device=device)

    if mode == 'test':
        model_name = sys.argv[2]
        test_model(model_name=model_name, device=device)

    if mode == 'prepare_user_data':
        user = sys.argv[2]
        n_tweets = int(sys.argv[3])
        prepare_user_data(user, n_tweets)


