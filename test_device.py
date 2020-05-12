import torch
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print('device is:', device)

import torchtext
from torchtext.data.utils import get_tokenizer

################### prepare the data ###################

def prepare_data(batch_size, eval_batch_size, device):
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                #tokenize="spacy",
                                init_token='<sos>', eos_token='<eos>', lower=True,
                                unk_token='<unk>', pad_token='<pad>')

    train_text, val_txt, test_txt = torchtext.data.TabularDataset.splits(root='data', path='data/@realDonaldTrump/',
                                                                         train='train.csv', validation='val.csv',
                                                                         test='test.csv', fields=[('index', None),
                                                                                                  ('text', TEXT)],
                                                                         format='csv', skip_header=True)
    #train_text, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_text)

    def batchify(data, batch_size):

        #print(type(data.examples[0].text))
        #print(data.examples[0].text)
        data = TEXT.numericalize([data.examples[0].text])

        n_batches = data.size(0) // batch_size
        data = data.narrow(0, 0, n_batches * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        return data.to(device)

    train_data = batchify(train_text, batch_size)
    val_data = batchify(val_txt, eval_batch_size)
    test_data = batchify(test_txt, eval_batch_size)

    n_words = len(TEXT.vocab.stoi)

    return train_data, val_data, test_data, n_words, TEXT

train_data, val_data, test_data, n_words, TEXT = prepare_data(20, 10, device)

print(n_words, 'words')

print(train_data.size())
sentence_ids = train_data[0,0:50]
print(sentence_ids)

sentence = list(map(lambda x: TEXT.vocab.itos[x], sentence_ids))
print(sentence)