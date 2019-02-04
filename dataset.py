import spacy
import torch
from torchtext import data
import dill
import pickle


def get_data(batch_size, device):

    spacy_en = spacy.load('en')

    def tokenizer(text): # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = data.LabelField(dtype=torch.float)

    tv_datafields = [("Text", TEXT), ("Label", LABEL)]

    train_data, val_data, test_data = data.TabularDataset.splits(
        path="/data/Andrew_Externs_Headnotes_20190103/blank_0.5_validation_0.01_binary_normpunct_january092019/", # the root directory where the data lies
        train='train_cat_sample', validation='valid_cat', test='test_sample',
        format='csv',
        skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
        fields=tv_datafields)

    TEXT.build_vocab(train_data, max_size=100000, vectors="glove.6B.100d")
    LABEL.build_vocab(train_data)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=BATCH_SIZE,
        device=device, sort_key=lambda x: len(x.Text))

    return train_iterator, valid_iterator, test_iterator
