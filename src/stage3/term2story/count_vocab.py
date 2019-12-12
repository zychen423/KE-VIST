import os
import spacy
import pickle
import json
import argparse
from collections import Counter
import numpy as np
import re
from transformer import Constants
nlp = spacy.load("en_core_web_sm", disable=['tagger','parser','ner', 'vector'])
from spacy.symbols import ORTH
nlp.tokenizer.add_special_case(u'[female]', [{ORTH: u'[female]'}])
nlp.tokenizer.add_special_case(u'[male]', [{ORTH: u'[male]'}])
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
    # dialog = json.load(open(text[0], 'r'))
    counter = Counter()
    dialog = json.load(open(text))['output_stories']
    #dialog = json.load(open(text))
    for i, dia in enumerate(dialog):
    #for i, dia in enumerate(dialog.values()):
        if i % 100 == 0 :
            print(i)
        u = dia['story_text_normalized']
        #u = dia
        sentence = u.lower()
        tokens = nlp.tokenizer(sentence)
        tokens = [t.text for t in tokens]
        #print(tokens)
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # Creates a vocab wrapper and add some special .
    vocab = Vocabulary()
    word_count = {}
    for word, cnt in counter.items():
        word_count[word] = cnt
    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    for word, idx in vocab.word2idx.items():
        if word in word_count.keys():
            count = word_count[word]
            vocab.word_count.append(int(count))
        else:
            vocab.word_count.append(int(1))

    total_tokens = sum(vocab.word_count)
    rate_word_count = [ i/total_tokens for i in vocab.word_count]
    out_calmulative=[]
    for i in range(1,101):
        count=0
        for c in rate_word_count:
            if c <= i*0.00001:
                print(c, i*0.00001, count, len(vocab))
                count+=1
        out_calmulative.append(count/len(vocab))
    print(out_calmulative)
def main(args):
    vocab = build_vocab(text=args.caption_path, threshold=args.threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default="../data/ROC_train.json", help='path for train annotation file')
    parser.add_argument('--threshold', type=int, default=0,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
