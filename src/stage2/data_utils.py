from collections import defaultdict
import torch
import random
import os
from torch.utils.data import Dataset, DataLoader, Subset
import json


NUM_WORKERS = 0


class Tokenizer():
    def __init__(self, corpus):
        term2appear = defaultdict(int)
        for termset_seq in corpus:
            for termset in termset_seq:
                for term in termset:
                    term2appear[term] += 1
        terms = [k for k, v in sorted(
            term2appear.items(), key=lambda item: item[1])]
        self.term2id = {'PAD': 0, 'BOS': 1}
        for term in terms:
            if term not in self.term2id:
                self.term2id[term] = len(self.term2id)
        self.term2id['UNK'] = len(self.term2id)
        self.vocab_size = len(self.term2id)

    def tokenize(self, corpus):
        tokened_corpus = []
        for termset_seq in corpus:
            new_list = []
            for termset in termset_seq:
                for term in termset:
                    if term in self.term2id:
                        new_list.append(self.term2id[term])
                    else:
                        new_list.append(self.term2id['UNK'])
            tokened_corpus.append(new_list)
        return tokened_corpus


class FlattenDataset(Dataset):
    def __init__(self, corpus, tokenizer):
        self.datas = tokenizer.tokenize(corpus)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]


def split_dataset(dataset):
    indicies = list(range(len(dataset)))
    random.shuffle(indicies)
    train_indicies = indicies[: int(0.8*len(indicies))]
    dev_indicies = indicies[int(0.8*len(indicies)): int(0.9*len(indicies))]
    test_indicies = indicies[int(0.9*len(indicies)):]
    train_dataset = Subset(dataset, train_indicies)
    dev_dataset = Subset(dataset, dev_indicies)
    test_dataset = Subset(dataset, test_indicies)
    return train_dataset, dev_dataset, test_dataset


def right_shift(batch, BOS):
    return [[BOS] + x[:-1] for x in batch]


def padding(batch, PAD):
    maxlen = max([len(x) for x in batch])
    return [x+[PAD]*(maxlen-len(x)) for x in batch]


def lm_collate(batch, PAD):
    # input: BOS, term0, term1
    # target: term0, term1, term2
    input = right_shift(batch, PAD+1)
    lens = [len(x) for x in input]
    input = padding(input, PAD)
    output = padding(batch, PAD)
    input = torch.LongTensor(input)
    lens = torch.LongTensor(lens)
    output = torch.LongTensor(output)
    return input, lens, output
