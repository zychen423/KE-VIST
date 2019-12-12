import os
import nltk
import pickle
import json
import argparse
from collections import Counter
import numpy as np
import re
from transformer import Constants
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_count = []

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(text, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    dialog = json.load(open(text[0]))
    print(len(dialog))
    for i, dia in enumerate(dialog):
        if i % 5000 == 0 :
            print(i)
            print(dia['coref_mapped_seq'])
        for _, u in enumerate(dia['coref_mapped_seq']):
            sentence = u
            counter.update(sentence)
    dialog = json.load(open(text[1]))
    print(len(dialog))
    for i, dia in enumerate(dialog):
        if i % 5000 == 0 :
            print(i)
            print(dia['coref_mapped_seq'])
        sentence = dia['coref_mapped_seq']
        counter.update(sentence)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # Creates a vocab wrapper and add some special .
    vocab = Vocabulary()
    word_count = {}
    for word, cnt in counter.items():
        word_count[word] = cnt
    vocab.add_word(Constants.PAD_WORD)
    vocab.add_word(Constants.UNK_WORD)
    vocab.add_word(Constants.BOS_WORD)
    vocab.add_word(Constants.BOS_WORD2)
    vocab.add_word(Constants.BOS_WORD3)
    vocab.add_word(Constants.BOS_WORD4)
    vocab.add_word(Constants.BOS_WORD5)
    vocab.add_word(Constants.EOS_WORD)
    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    for word, idx in vocab.word2idx.items():
        if word in word_count.keys():
            count = word_count[word]
            vocab.word_count.append(count)
        else:
            vocab.word_count.append(int(1))
    return vocab

def main(args):
    if not os.path.exists(args.vocab_dir):
        os.makedirs(args.vocab_dir)
        print("Make Data Directory")
    vocab = build_vocab(text=[args.caption_path, args.caption_path_1],
                        threshold=args.threshold)
    vocab_path = os.path.join(args.vocab_dir, 'term_vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: %d" %len(vocab))
    #print(vocab.word2idx)
    #print(vocab.word_count)
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default="../data/ROC/ROC_train.json",
                        help='path for train annotation file')
    parser.add_argument('--caption_path_1', type=str,
                        default="../data/VIST/VIST_coref_nos_mapped_frame_noun_train.json",
                        help='path for train annotation file')
    parser.add_argument('--vocab_dir', type=str, default='../data/term2story_vocabs/',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
