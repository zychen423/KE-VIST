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

def build_glove_voc(threshold, vocab, paragraph):
    data_path = '/corpus/glove/pretrained_vector/english/glove.42B.300d.{}'
    with open(data_path.format('json'),'r', encoding='utf-8') as f:
        glove = json.load(f, encoding='utf-8')

    count = 0
    word2vec = {}
    if paragraph:
        weight_matrix = np.random.uniform(-0.5, 0.5, size=(threshold+6,300))
    else:
        weight_matrix = np.random.uniform(-0.5, 0.5, size=(threshold+4,300))

    with open(data_path.format('txt'),'r', encoding='utf8') as f:
        for line in f:
            l = line.strip().split()
            word = l[0]
            if vocab(word) != 3:
                weight_matrix[vocab(word),:] = np.asarray(list(map(float, l[1:])))

            count += 1


    return weight_matrix

def clean_str(string):
    string = string.lower()
    string = re.sub(u"(\u2018|\u2019)", "'", string)
    string = re.sub(u'(\u201c|\u201d)', '"', string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\"s", " \'s", string)
    string = re.sub(r"\"ve", " \'ve", string)
    string = re.sub(r"n\"t", " n\'t", string)
    string = re.sub(r"\"re", " \'re", string)
    string = re.sub(r"\"d", " \'d", string)
    string = re.sub(r"\"ll", " \'ll", string)
    string = re.sub(r"\"m", " \'m", string)
    string = re.sub(r"\."," .", string)
    string = re.sub(r"\!"," !", string)
    string = re.sub(r"\,"," ,", string)
    #string = re.sub(r" "," ", string)


    return string

def build_vocab(text, threshold, coco):
    """Build a simple vocabulary wrapper."""
    # dialog = json.load(open(text[0], 'r'))
    counter = Counter()

    print(text[0])
    dialog = json.load(open(text[0]))
    print(len(dialog))
    for i, dia in enumerate(dialog):
        if i % 5000 == 0 :
            print(i)
        for _, u in enumerate(dia['ner_story']):
            sentence = u.lower()
            tokens = nlp.tokenizer(sentence)
            tokens = [t.text for t in tokens]
            #print(tokens)
            counter.update(tokens)
    print(text[1])
    dialog = json.load(open(text[1]))
    print(len(dialog))
    for i, dia in enumerate(dialog):
        if i % 5000 == 0 :
            print(i)
        sentence = dia['text']
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
            vocab.word_count.append(1/count)
        else:
            vocab.word_count.append(int(1))
    return vocab

def main(args):
    if not os.path.exists(args.vocab_dir):
        os.makedirs(args.vocab_dir)
        print("Make Data Directory")
    vocab = build_vocab(text=[args.caption_path, args.caption_path_1],
                        threshold=args.threshold, coco=args.coco)
    #W = build_glove_voc(len(vocab), vocab, args.paragraph)
    vocab_path = os.path.join(args.vocab_dir, 'story_vocab.pkl')
    #weight_path = os.path.join(args.vocab_dir, 'W.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    #with open(weight_path, 'wb') as f:
    #    pickle.dump(W, f)

    print("Total vocabulary size: %d" %len(vocab))
    #print(vocab.word2idx)
    #print(vocab.word_count)
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default="../data/ROC/ROC_train.json", help='path for train annotation file')
    parser.add_argument('--caption_path_1', type=str,
                        default="../data/VIST/VIST_coref_nos_mapped_frame_noun_train.json", help='path for train annotation file')
    parser.add_argument('--vocab_dir', type=str, default='../data/term2story_vocabs/',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5,
                        help='minimum word count threshold')
    parser.add_argument('--coco', action='store_true', default=False,
                        help='minimum word count threshold')
    parser.add_argument('--parse', action='store_true', default=False,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
