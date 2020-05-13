import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools

###################### USED IN DATA PREPARATION ######################
import torchtext
from torchtext.data.utils import get_tokenizer
import re
######################################################################

import time
import pandas as pd
import numpy as np
import random

import json
import os
import csv
import sys

class LambdaLR():
    def __init__(self, n_epochs, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = 0
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data_src = []
        self.data_lens = []

    def push_and_pop(self, src, lens):
        to_return_src = []
        to_return_lens = []
        src = src.transpose(0, 1)
        for src_element, lens_element in zip(src, lens):
            src_element = torch.unsqueeze(src_element, 0)
            lens_element = torch.unsqueeze(lens_element, 0)
            if len(self.data_src) < self.max_size:
                self.data_src.append(src_element)
                self.data_lens.append(lens_element)

                to_return_src.append(src_element)
                to_return_lens.append(lens_element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return_src.append(self.data_src[i].clone())
                    to_return_lens.append(self.data_lens[i].clone())

                    self.data_src[i] = src_element
                    self.data_lens[i] = lens_element
                else:
                    to_return_src.append(src_element)
                    to_return_lens.append(lens_element)
        return Variable(torch.cat(to_return_src).transpose(0, 1)), Variable(torch.cat(to_return_lens))



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
                 src_encoder, dropout=0.5, device='cuda', max_len=50, pad=0, sos=1, eos=2):
        super(Generator, self).__init__()

        self.device = device
        self.model_type = 'Transformer'
        self.sos = sos
        self.eos = eos
        self.max_len = max_len

        self.embedded_size = embedded_size
        self.input_vocab_size = input_vocab_size
        self.src_encoder = src_encoder.to(device)

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

def train_model(model_name, user_A, user_B, n_epochs, decay_epoch, device):

    ################### prepare the data ###################

    def prepare_data(user, batch_size, max_len, vocab_itos, vocab_stoi, device):
        train_df = pd.read_csv('data/' + user + '/train.csv')
        val_df = pd.read_csv('data/' + user + '/val.csv')
        test_df = pd.read_csv('data/' + user + '/test.csv')

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

        return train_data, val_data, test_data, train_lens, val_lens, test_lens
    ################### training settings ###################

    batch_size = 20
    max_len = 50

    vocab_itos = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
    vocab_stoi = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}

    (A_train_data, A_val_data, A_test_data,
     A_train_lens, A_val_lens, A_test_lens) = prepare_data(user_A, batch_size=batch_size, max_len=max_len,
                                                           vocab_itos=vocab_itos, vocab_stoi=vocab_stoi, device=device)

    (B_train_data, B_val_data, B_test_data,
     B_train_lens, B_val_lens, B_test_lens) = prepare_data(user_B, batch_size=batch_size, max_len=max_len,
                                                           vocab_itos=vocab_itos, vocab_stoi=vocab_stoi, device=device)

    A_train_data = A_train_data.to(device)
    B_train_data = B_train_data.to(device)

    n_batches = A_train_data.size(0)

    vocab_size = len(vocab_itos)

    print('vocab size:', vocab_size)

    G_embedded_size = 256
    G_n_heads = 8
    G_n_hidden = 1024
    G_n_layers = 6
    G_dropout = 0.1

    shared_encoder = nn.Embedding(vocab_size, G_embedded_size, padding_idx=0).to(device)

    netG_A2B = Generator(input_vocab_size=vocab_size, output_vocab_size=vocab_size, embedded_size=G_embedded_size,
                         n_heads=G_n_heads, n_hidden=G_n_hidden, n_layers=G_n_layers, dropout=G_dropout,
                         src_encoder=shared_encoder, device=device).to(device)

    netG_B2A = Generator(input_vocab_size=vocab_size, output_vocab_size=vocab_size, embedded_size=G_embedded_size,
                         n_heads=G_n_heads, n_hidden=G_n_hidden, n_layers=G_n_layers, dropout=G_dropout,
                         src_encoder=shared_encoder, device=device).to(device)

    D_embedded_size = 32
    D_n_heads = 1
    D_n_hidden = 64
    D_n_layers = 1
    D_dropout = 0.1

    netD_A = Discriminator(vocab_size=vocab_size, embedded_size=D_embedded_size, n_heads=D_n_heads,
                                  n_hidden=D_n_hidden, n_layers=D_n_layers, dropout=D_dropout,
                                  device=device).to(device)

    netD_B = Discriminator(vocab_size=vocab_size, embedded_size=D_embedded_size, n_heads=D_n_heads,
                           n_hidden=D_n_hidden, n_layers=D_n_layers, dropout=D_dropout,
                           device=device).to(device)

    ################# training loop #################

    # losses

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers

    lr = 0.0002
    betas = (0.5, 0.999)

    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=betas)
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=betas)
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=betas)

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, decay_epoch).step)

    # Inputs & targets memory allocation

    LongTensor = torch.cuda.LongTensor if device == 'cuda' else torch.LongTensor
    FloatTensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
    A_src = LongTensor(max_len, batch_size)
    A_lens = LongTensor(batch_size)
    B_src = LongTensor(max_len, batch_size)
    B_lens = LongTensor(batch_size)
    target_real = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    n_fixed = 5

    fixed_A_src = A_train_data[0, :, :n_fixed]
    fixed_A_lens = A_train_lens[0, :n_fixed]
    A_tweets_list = []

    fixed_B_src = B_train_data[0, :, :n_fixed]
    fixed_B_lens = B_train_lens[0, :n_fixed]
    B_tweets_list = []

    G_losses = []
    D_losses = []

    # Training
    print("Starting Training Loop...")
    for epoch in range(n_epochs):
        for batch_idx in range(n_batches):
            real_A_src = Variable(A_src.copy_(A_train_data[batch_idx]))
            real_A_lens = Variable(A_lens.copy_(A_train_lens[batch_idx]))

            real_B_src = Variable(B_src.copy_(B_train_data[batch_idx]))
            real_B_lens = Variable(B_lens.copy_(B_train_lens[batch_idx]))

            ##############################################################
            #                     Update Generators                      #
            ##############################################################

            optimizer_G.zero_grad()

            # Identity loss: G_A2B(B) = B and vice versa
            same_B_src, same_B_lens = netG_A2B(real_B_src, real_B_lens)
            embedded_same_B = shared_encoder(same_B_src)
            embedded_real_B = shared_encoder(real_B_src)
            loss_identity_B = criterion_identity(embedded_same_B, embedded_real_B) * 5.0

            same_A_src, same_A_lens = netG_B2A(real_A_src, real_A_lens)
            embedded_same_A = shared_encoder(same_A_src)
            embedded_real_A = shared_encoder(real_A_src)
            loss_identity_A = criterion_identity(embedded_same_A, embedded_real_A) * 5.0

            # GAN loss
            fake_B_src, fake_B_lens = netG_A2B(real_A_src, real_A_lens)
            pred_fake = netD_B(fake_B_src, fake_B_lens).view(-1)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A_src, fake_A_lens = netG_B2A(real_B_src, real_B_lens)
            pred_fake = netD_A(fake_A_src, fake_A_lens).view(-1)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A_scr, recovered_A_lens = netG_B2A(fake_B_src, fake_B_lens)
            embedded_recovered_A = shared_encoder(recovered_A_scr)
            loss_cycle_ABA = criterion_cycle(embedded_recovered_A, embedded_real_A) * 10.0

            recovered_B_scr, recovered_B_lens = netG_A2B(fake_A_src, fake_A_lens)
            embedded_recovered_B = shared_encoder(recovered_B_scr)
            loss_cycle_BAB = criterion_cycle(embedded_recovered_B, embedded_real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()

            ##############################################################
            #                   Update Discriminator A                   #
            ##############################################################

            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A_src, real_A_lens).view(-1)
            D_A_x = pred_real.mean().item()

            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A_src, fake_A_lens = fake_A_buffer.push_and_pop(fake_A_src, fake_A_lens)
            pred_fake = netD_A(fake_A_src.detach(), fake_A_lens.detach()).view(-1)
            D_A_G_x = pred_fake.mean().item()

            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            ##############################################################
            #                   Update Discriminator B                   #
            ##############################################################

            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B_src, real_B_lens).view(-1)
            D_B_x = pred_real.mean().item()

            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B_src, fake_B_lens = fake_B_buffer.push_and_pop(fake_B_src, fake_B_lens)
            pred_fake = netD_B(fake_B_src.detach(), fake_B_lens.detach()).view(-1)
            D_B_G_x = pred_fake.mean().item()

            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()

            ##############################################################
            #                   Output Training Stats                    #
            ##############################################################

            D_x = (D_A_x + D_B_x) / 2
            D_G_x = (D_A_G_x + D_B_G_x) / 2

            loss_D = loss_D_A + loss_D_B

            if batch_idx % 20 == 0 or batch_idx == n_batches - 1:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): '
                      '%.4f\tD(G(x)): %.4f' % (epoch, n_epochs, batch_idx, n_batches, loss_D.item(),
                                                      loss_G.item(), D_x, D_G_x))

            D_losses.append(loss_D.item())
            G_losses.append(loss_G.item())

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        with torch.no_grad():
            fake_B_src, fake_B_lens = netG_A2B(fixed_A_src, fixed_A_lens)
            fake_A_src, fake_A_lens = netG_B2A(fixed_B_src, fixed_B_lens)
        B_tweets_list.append(tokens_to_sentences(fake_B_src, fake_B_lens, vocab_itos))
        A_tweets_list.append(tokens_to_sentences(fake_A_src, fake_A_lens, vocab_itos))

    ########### show translation evolution ###########

    with open("models/" + model_name + '_evolution.txt', 'w') as f:

        f.write('=' * 100 + '\n')
        f.write('Source tweets examples from ' + user_A +'\n')
        f.write('=' * 100 + '\n')

        fixed_sentences = tokens_to_sentences(fixed_A_src, fixed_A_lens, vocab_itos)
        for sentence in fixed_sentences:
            f.write(" ".join(sentence) + '\n')
        f.write('='*100 + '\n')

        f.write('=' * 100 + '\n')
        f.write('Translated tweets examples from ' + user_A + ' to ' + user_B + '\n')
        f.write('=' * 100 + '\n')

        for epoch, sentences in enumerate(B_tweets_list):
            f.write('epoch:' + str(epoch) + '\n')
            for sentence in sentences:
                f.write(" ".join(sentence) + '\n')
            f.write('='*100 + '\n')

        f.write('\n')

        f.write('=' * 100 + '\n')
        f.write('Source tweets examples from ' + user_B + '\n')
        f.write('=' * 100 + '\n')

        fixed_sentences = tokens_to_sentences(fixed_B_src, fixed_B_lens, vocab_itos)
        for sentence in fixed_sentences:
            f.write(" ".join(sentence) + '\n')
        f.write('=' * 100 + '\n')

        f.write('=' * 100 + '\n')
        f.write('Translated tweets examples from ' + user_B + ' to ' + user_A + '\n')
        f.write('=' * 100 + '\n')

        for epoch, sentences in enumerate(A_tweets_list):
            f.write('epoch:' + str(epoch) + '\n')
            for sentence in sentences:
                f.write(" ".join(sentence) + '\n')
            f.write('=' * 100 + '\n')

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

    netG_A2B_params = netG_A2B.state_dict()

    netG_A2B_kwargs = {'input_vocab_size': vocab_size, 'output_vocab_size': vocab_size,
                       'embedded_size': G_embedded_size, 'n_heads': G_n_heads, 'n_hidden': G_n_hidden,
                       'n_layers': G_n_layers, 'dropout': G_dropout}

    save_model(model_name + '_netG_A2B', netG_A2B_params, netG_A2B_kwargs)

    netG_B2A_params = netG_B2A.state_dict()

    netG_B2A_kwargs = {'input_vocab_size': vocab_size, 'output_vocab_size': vocab_size,
                       'embedded_size': G_embedded_size, 'n_heads': G_n_heads, 'n_hidden': G_n_hidden,
                       'n_layers': G_n_layers, 'dropout': G_dropout}

    save_model(model_name + '_netG_B2A', netG_B2A_params, netG_B2A_kwargs)

    netD_A_params = netD_A.state_dict()

    netD_A_kwargs = {'vocab_size':vocab_size, 'embedded_size': D_embedded_size, 'n_heads': D_n_heads,
                     'n_hidden': D_n_hidden, 'n_layers': D_n_layers, 'dropout': D_dropout}

    save_model(model_name + '_netD_A', netD_A_params, netD_A_kwargs)

    netD_B_params = netD_B.state_dict()

    netD_B_kwargs = {'vocab_size': vocab_size, 'embedded_size': D_embedded_size, 'n_heads': D_n_heads,
                     'n_hidden': D_n_hidden, 'n_layers': D_n_layers, 'dropout': D_dropout}

    save_model(model_name + '_netD_B', netD_B_params, netD_B_kwargs)

    return netG_A2B, netG_B2A, netD_A, netD_B

def save_model(model_name, model_params, model_kwargs):

    params_path = 'models/' + model_name + '_params.pt'

    torch.save(model_params, params_path)

    kwargs_path = 'models/' + model_name + '_kwargs.json'

    with open(kwargs_path, 'w') as kwargs_file:
        json.dump(model_kwargs, kwargs_file, sort_keys=True, indent=4)

    print(model_name, 'saved')


def load_model(model_name, device):

    netG_A2B_kwargs_path = 'models/' + model_name + '_netG_A2B_kwargs.json'
    with open(netG_A2B_kwargs_path, 'r') as netG_A2B_kwargs_file:
        netG_A2B_kwargs = json.load(netG_A2B_kwargs_file)

    netG_A2B = Generator(**netG_A2B_kwargs).to(device)

    netG_A2B_params_path = 'models/' + model_name + '_netG_A2B_params.pt'

    netG_A2B.load_state_dict(torch.load(netG_A2B_params_path))
    netG_A2B.eval()

    netG_B2A_kwargs_path = 'models/' + model_name + '_netG_B2A_kwargs.json'
    with open(netG_B2A_kwargs_path, 'r') as netG_B2A_kwargs_file:
        netG_B2A_kwargs = json.load(netG_B2A_kwargs_file)

    netG_B2A = Generator(**netG_B2A_kwargs).to(device)

    netG_B2A_params_path = 'models/' + model_name + '_netG_B2A_params.pt'

    netG_B2A.load_state_dict(torch.load(netG_B2A_params_path))
    netG_B2A.eval()

    netD_A_kwargs_path = 'models/' + model_name + '_netD_A_kwargs.json'
    with open(netD_A_kwargs_path, 'r') as netD_A_kwargs_file:
        netD_A_kwargs = json.load(netD_A_kwargs_file)

    netD_A = Discriminator(**netD_A_kwargs).to(device)

    netD_A_params_path = 'models/' + model_name + '_netD_A_params.pt'

    netD_A.load_state_dict(torch.load(netD_A_params_path))
    netD_A.eval()

    netD_B_kwargs_path = 'models/' + model_name + '_netD_B_kwargs.json'
    with open(netD_B_kwargs_path, 'r') as netD_B_kwargs_file:
        netD_B_kwargs = json.load(netD_B_kwargs_file)

    netD_B = Discriminator(**netD_B_kwargs).to(device)

    netD_B_params_path = 'models/' + model_name + '_netD_B_params.pt'

    netD_B.load_state_dict(torch.load(netD_B_params_path))
    netD_B.eval()

    print(model_name, 'loaded with device', str(generator.device))

    return netG_A2B, netG_B2A, netD_A, netD_B


def test_model(model_name, device):

    netG_A2B, netG_B2A, netD_A, netD_B = load_model(model_name, device)

    print(netG_A2B.state_dict())
    print(netG_B2A.state_dict())
    print(netD_A.state_dict())
    print(netD_B.state_dict())

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
    print('device is:', device)

    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    #random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    random.seed(manualSeed)

    mode = sys.argv[1]

    if mode == 'train':
        model_name, user_A, user_B, n_epochs, decay_epoch = (sys.argv[2], sys.argv[3], sys.argv[4],
                                                             int(sys.argv[5]), int(sys.argv[6]))
        netG_A2B, netG_B2A, netD_A, netD_B = train_model(model_name=model_name, user_A=user_A, user_B=user_B,
                                                         n_epochs=n_epochs, decay_epoch=decay_epoch, device=device)

    if mode == 'test':
        model_name = sys.argv[2]
        test_model(model_name=model_name, device=device)

    if mode == 'prepare_user_data':
        user = sys.argv[2]
        n_tweets = int(sys.argv[3])
        prepare_user_data(user, n_tweets)