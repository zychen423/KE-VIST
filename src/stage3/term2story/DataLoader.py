import torch
from torch.utils.data.dataset import Dataset
import json
from build_story_vocab import Vocabulary
from transformer import Constants
import spacy
nlp = spacy.load("en_core_web_sm", disable=['tagger','parser','ner', 'vector'])
from spacy.symbols import ORTH
nlp.tokenizer.add_special_case(u'[female]', [{ORTH: u'[female]'}])
nlp.tokenizer.add_special_case(u'[male]', [{ORTH: u'[male]'}])


class ROCDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path, is_verb=True):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9
        self.max_sentence_len = 24
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab

    def __getitem__(self, index):
        frame = []
        story = []
        dialog = self.dialogs[index]
        for i in range(5):
            sentence = []
            tmp_frame = []
            Frame = dialog['coref_mapped_seq'][i]
            description = dialog['ner_story'][i].lower()
            tokens = nlp.tokenizer(description)
            sentence.extend([self.story_vocab(token.text) for token in tokens])
            tmp_frame.extend([self.frame_vocab(F) for F in Frame])

            if len(sentence) > self.max_sentence_len-1:
                sentence = sentence[:self.max_sentence_len]
            if len(tmp_frame) > self.max_term_len-1:
                tmp_frame = tmp_frame[:self.max_term_len]

            frame.append(tmp_frame)
            story.append(sentence)

        S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []
        for i, (s, f ) in enumerate(zip(story, frame)):
            S.append(Constants.BOSs[0])
            F.append(Constants.BOSs[0])
            S.extend(s)
            F.extend(f)
            S_sen_pos.extend([i+1]*(len(s)+1))
            S_word_pos.extend([i+1 for i in range(len(s)+1)])
            F_sen_pos.extend([i+1]*(len(f)+1))
            F_word_pos.extend([i+1 for i in range(len(f)+1)])
        #print(len(F), len(F_sen_pos))
        assert len(S) == len(S_sen_pos)
        assert len(S) == len(S_word_pos)
        assert len(F) == len(F_sen_pos)
        assert len(F) == len(F_word_pos)
        return S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos

    def __len__(self):
        return len(self.dialogs)

class ROCAddTermsetDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path, is_verb=True):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9
        self.max_sentence_len = 24
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab

    def __getitem__(self, index):
        dialog = self.dialogs[index]
        wS, wS_sen_pos, wS_word_pos, wF, wF_sen_pos, wF_word_pos = [], [], [], [], [], []
        for window in range(len(dialog)-4):
            frame = []
            story = []
            for i in range(5):
                sentence = []
                tmp_frame = []
                Frame = dialog[i+window]['predicted_term_seq']
                description = dialog[i+window]['text'].lower()
                tokens = nlp.tokenizer(description)
                sentence.extend([self.story_vocab(token.text) for token in tokens])
                tmp_frame.extend([self.frame_vocab(F) for F in Frame])

                if len(sentence) > self.max_sentence_len-1:
                    sentence = sentence[:self.max_sentence_len]
                if len(tmp_frame) > self.max_term_len-1:
                    tmp_frame = tmp_frame[:self.max_term_len]

                frame.append(tmp_frame)
                story.append(sentence)
            S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []
            for i, (s, f ) in enumerate(zip(story, frame)):
                S.append(Constants.BOSs[0])
                F.append(Constants.BOSs[0])
                S.extend(s)
                F.extend(f)
                S_sen_pos.extend([i+1]*(len(s)+1))
                S_word_pos.extend([i+1 for i in range(len(s)+1)])
                F_sen_pos.extend([i+1]*(len(f)+1))
                F_word_pos.extend([i+1 for i in range(len(f)+1)])
            #print(len(F), len(F_sen_pos))
            assert len(S) == len(S_sen_pos)
            assert len(S) == len(S_word_pos)
            assert len(F) == len(F_sen_pos)
            assert len(F) == len(F_word_pos)
            wS.append(S)
            wS_sen_pos.append(S_sen_pos)
            wS_word_pos.append(S_word_pos)
            wF.append(F)
            wF_sen_pos.append(F_sen_pos)
            wF_word_pos.append(F_word_pos)
        return wS, wS_sen_pos, wS_word_pos, wF, wF_sen_pos, wF_word_pos

    def __len__(self):
        return len(self.dialogs)
    
class ROCAddTermsetDataset_SixSentence(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path, is_verb=True):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9
        self.max_sentence_len = 24
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab

    def __getitem__(self, index):
        dialog = self.dialogs[index]
        wS, wS_sen_pos, wS_word_pos, wF, wF_sen_pos, wF_word_pos = [], [], [], [], [], []
        frame = []
        story = []
        for window in range(len(dialog)-4):
            for i in range(5):
                sentence = []
                tmp_frame = []
                Frame = dialog[i+window]['predicted_term_seq']
                description = dialog[i+window]['text'].lower()
                tokens = nlp.tokenizer(description)
                sentence.extend([self.story_vocab(token.text) for token in tokens])
                tmp_frame.extend([self.frame_vocab(F) for F in Frame])

                if len(sentence) > self.max_sentence_len-1:
                    sentence = sentence[:self.max_sentence_len]
                if len(tmp_frame) > self.max_term_len-1:
                    tmp_frame = tmp_frame[:self.max_term_len]
                    
                if window == 0:
                    frame.append(tmp_frame)
                    story.append(sentence)
                else:
                    if i == 4:
                        frame.append(tmp_frame)
                        story.append(sentence)
        S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []
        for i, (s, f ) in enumerate(zip(story, frame)):
            S.append(Constants.BOSs[0])
            F.append(Constants.BOSs[0])
            S.extend(s)
            F.extend(f)
            S_sen_pos.extend([i+1]*(len(s)+1))
            S_word_pos.extend([i+1 for i in range(len(s)+1)])
            F_sen_pos.extend([i+1]*(len(f)+1))
            F_word_pos.extend([i+1 for i in range(len(f)+1)])
        #print(len(F), len(F_sen_pos))
        assert len(S) == len(S_sen_pos)
        assert len(S) == len(S_word_pos)
        assert len(F) == len(F_sen_pos)
        assert len(F) == len(F_word_pos)
        wS.append(S)
        wS_sen_pos.append(S_sen_pos)
        wS_word_pos.append(S_word_pos)
        wF.append(F)
        wF_sen_pos.append(F_sen_pos)
        wF_word_pos.append(F_word_pos)
        return wS, wS_sen_pos, wS_word_pos, wF, wF_sen_pos, wF_word_pos

    def __len__(self):
        return len(self.dialogs)
    

class ROCAddDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path, is_verb=True):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_sentence_len = 24
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab
        self.is_verb = is_verb

    def __getitem__(self, index):
        frame = []
        story = []
        for i in range(index, index+5):
            sentence = []
            tmp_frame = []
            Frame = self.dialogs[i]['predicted_term_seq']
            description = self.dialogs[i]['text'].lower()
            tokens = nlp.tokenizer(description)
            sentence.extend([self.story_vocab(token.text) for token in tokens])
            tmp_frame.extend([self.frame_vocab(F) for F in Frame])

            if len(sentence) > self.max_sentence_len-1:
                sentence = sentence[:self.max_sentence_len]
            if len(tmp_frame) > self.max_term_len-1:
                tmp_frame = tmp_frame[:self.max_term_len]

            frame.append(tmp_frame)
            story.append(sentence)

        S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []
        for i, (s, f ) in enumerate(zip(story, frame)):
            S.append(Constants.BOSs[0])
            F.append(Constants.BOSs[0])
            S.extend(s)
            F.extend(f)
            S_sen_pos.extend([i+1]*(len(s)+1))
            S_word_pos.extend([i+1 for i in range(len(s)+1)])
            F_sen_pos.extend([i+1]*(len(f)+1))
            F_word_pos.extend([i+1 for i in range(len(f)+1)])
        #print(len(F), len(F_sen_pos))
        assert len(S) == len(S_sen_pos)
        assert len(S) == len(S_word_pos)
        assert len(F) == len(F_sen_pos)
        assert len(F) == len(F_word_pos)
        return S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos

    def __len__(self):
        return len(self.dialogs)-4

class VISTDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9
        self.max_sentence_len = 24
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab


    def __getitem__(self, index):
        frame = []
        story = []
        for idx in range(index*5,index*5+5):
            dialog = self.dialogs[idx]
            sentence = []
            tmp_frame = []
            Frame = dialog['coref_mapped_seq']
            #Frame = dialog['predicted_term_seq']
            #description = sen['ner_description']
            description = dialog['text'].lower()
            #tokens = description.strip().split()
            tokens = nlp.tokenizer(description)
            sentence.extend([self.story_vocab(token.text) for token in tokens])
            tmp_frame.extend([self.frame_vocab(F) for F in Frame])

            if len(sentence) > self.max_sentence_len-1:
                sentence = sentence[:self.max_sentence_len]
            if len(tmp_frame) > self.max_term_len-1:
                tmp_frame = tmp_frame[:self.max_term_len]

            frame.append(tmp_frame)
            story.append(sentence)

        S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []
        for i, (s, f ) in enumerate(zip(story, frame)):
            #S.append(Constants.BOSs[i])
            #F.append(Constants.BOSs[i])
            S.append(Constants.BOSs[0])
            F.append(Constants.BOSs[0])
            S.extend(s)
            F.extend(f)
            S_sen_pos.extend([i+1]*(len(s)+1))
            S_word_pos.extend([i+1 for i in range(len(s)+1)])
            F_sen_pos.extend([i+1]*(len(f)+1))
            F_word_pos.extend([i+1 for i in range(len(f)+1)])
        #print(len(F), len(F_sen_pos))
        assert len(S) == len(S_sen_pos)
        assert len(S) == len(S_word_pos)
        assert len(F) == len(F_sen_pos)
        assert len(F) == len(F_word_pos)
        return S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos


    def __len__(self):
        return len(self.dialogs)//5

class PredictedVISTDataset(Dataset):
    def __init__(self,
                 story_vocab,
                 frame_vocab,
                 text_path):
        print(text_path)
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_term_len = 9
        self.max_sentence_len = 24
        self.story_vocab = story_vocab
        self.frame_vocab = frame_vocab


    def __getitem__(self, index):
        frame = []
        story = []
        for idx in range(index*5,index*5+5):
            dialog = self.dialogs[idx]
            sentence = []
            tmp_frame = []
            #Frame = dialog['coref_mapped_seq']
            Frame = dialog['predicted_term_seq']
            #description = sen['ner_description']
            description = dialog['text'].lower()
            #tokens = description.strip().split()
            tokens = nlp.tokenizer(description)
            sentence.extend([self.story_vocab(token.text) for token in tokens])
            tmp_frame.extend([self.frame_vocab(F) for F in Frame])

            if len(sentence) > self.max_sentence_len-1:
                sentence = sentence[:self.max_sentence_len]
            if len(tmp_frame) > self.max_term_len-1:
                tmp_frame = tmp_frame[:self.max_term_len]

            frame.append(tmp_frame)
            story.append(sentence)

        S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos = [], [], [], [], [], []
        for i, (s, f ) in enumerate(zip(story, frame)):
            #S.append(Constants.BOSs[i])
            #F.append(Constants.BOSs[i])
            S.append(Constants.BOSs[0])
            F.append(Constants.BOSs[0])
            S.extend(s)
            F.extend(f)
            S_sen_pos.extend([i+1]*(len(s)+1))
            S_word_pos.extend([i+1 for i in range(len(s)+1)])
            F_sen_pos.extend([i+1]*(len(f)+1))
            F_word_pos.extend([i+1 for i in range(len(f)+1)])
        #print(len(F), len(F_sen_pos))
        assert len(S) == len(S_sen_pos)
        assert len(S) == len(S_word_pos)
        assert len(F) == len(F_sen_pos)
        assert len(F) == len(F_word_pos)
        return S, S_sen_pos, S_word_pos, F, F_sen_pos, F_word_pos


    def __len__(self):
        return len(self.dialogs)//5


def ROC_collate_fn(data):

    #List of sentences and frames [B,]
    stories, s_sen_pos, s_word_pos, frames, f_sen_pos, f_word_pos = zip(*data)
    #print('ROC_collate_fn: s_sen_pos',s_sen_pos)
    #print('s_sen_pos[0]',s_sen_pos[0])
    lengths = [len(x)+1 for x in stories]
    #max_seq_len = max(lengths)
    max_seq_len = 126
    pad_stories = [s+ [Constants.EOS] + [Constants.PAD for _ in range(max_seq_len - len(s) - 1)] for s in stories]
    #stories_pos = [[pos_i+1 if w_i != 0 else 0
    #     for pos_i, w_i in enumerate(inst)] for inst in pad_stories]
    pad_s_sen_pos = [s + [Constants.PAD for _ in range(max_seq_len - len(s))] for s in s_sen_pos]
    stories_pos = [s+ [Constants.PAD for _ in range(max_seq_len - len(s))] for s in s_word_pos]


    frame_lengths = [len(x)+1 for x in frames]
    #max_frame_seq_len = max(frame_lengths)
    max_frame_seq_len = 51
    pad_frame = [s + [Constants.EOS] + [Constants.PAD for _ in range(max_frame_seq_len - len(s) - 1)] for s in frames]
    pad_f_sen_pos = [s + [Constants.PAD for _ in range(max_frame_seq_len - len(s))] for s in f_sen_pos]
    #frame_pos = [[pos_i+1 if w_i != 0 else 0
    #     for pos_i, w_i in enumerate(inst)] for inst in pad_frame]
    frame_pos = [s + [Constants.PAD for _ in range(max_frame_seq_len - len(s))] for s in f_word_pos]

    targets = torch.LongTensor(pad_stories).view(-1, max_seq_len)
    lengths = torch.LongTensor(lengths).view(-1,1)
    targets_pos = torch.LongTensor(stories_pos).view(-1, max_seq_len)
    targets_sen_pos = torch.LongTensor(pad_s_sen_pos).view(-1, max_seq_len)
    frame = torch.LongTensor(pad_frame).view(-1,max_frame_seq_len)
    frame_lengths = torch.LongTensor(frame_lengths).view(-1,1)
    frame_pos = torch.LongTensor(frame_pos).view(-1, max_frame_seq_len)
    frame_sen_pos = torch.LongTensor(pad_f_sen_pos).view(-1, max_frame_seq_len)
    return frame, frame_pos, frame_sen_pos, targets, targets_pos, targets_sen_pos

def ROC_added_termset_collate_fn(data):

    #List of sentences and frames [B,]
    stories, s_sen_pos, s_word_pos, frames, f_sen_pos, f_word_pos = zip(*data)
    stories, s_sen_pos, s_word_pos, frames, f_sen_pos, f_word_pos = stories[0], s_sen_pos[0], s_word_pos[0], frames[0], f_sen_pos[0], f_word_pos[0]
    lengths = [len(x)+1 for x in stories]
    #max_seq_len = max(lengths)
    max_seq_len = 126
    pad_stories = [s+ [Constants.EOS] + [Constants.PAD for _ in range(max_seq_len - len(s) - 1)] for s in stories]
    lengths = [len(x)+1 for x in pad_stories]
    #max_seq_len = max(lengths)
    #stories_pos = [[pos_i+1 if w_i != 0 else 0
    #     for pos_i, w_i in enumerate(inst)] for inst in pad_stories]
    pad_s_sen_pos = [s + [Constants.PAD for _ in range(max_seq_len - len(s))] for s in s_sen_pos]
    stories_pos = [s+ [Constants.PAD for _ in range(max_seq_len - len(s))] for s in s_word_pos]


    frame_lengths = [len(x)+1 for x in frames]
    #max_frame_seq_len = max(frame_lengths)
    max_frame_seq_len = 51
    pad_frame = [s + [Constants.EOS] + [Constants.PAD for _ in range(max_frame_seq_len - len(s) - 1)] for s in frames]
    pad_f_sen_pos = [s + [Constants.PAD for _ in range(max_frame_seq_len - len(s))] for s in f_sen_pos]
    #frame_pos = [[pos_i+1 if w_i != 0 else 0
    #     for pos_i, w_i in enumerate(inst)] for inst in pad_frame]
    frame_pos = [s + [Constants.PAD for _ in range(max_frame_seq_len - len(s))] for s in f_word_pos]

    targets = torch.LongTensor(pad_stories).view(-1, max_seq_len)
    lengths = torch.LongTensor(lengths).view(-1,1)
    targets_pos = torch.LongTensor(stories_pos).view(-1, max_seq_len)
    targets_sen_pos = torch.LongTensor(pad_s_sen_pos).view(-1, max_seq_len)
    frame = torch.LongTensor(pad_frame).view(-1,max_frame_seq_len)
    frame_lengths = torch.LongTensor(frame_lengths).view(-1,1)
    frame_pos = torch.LongTensor(frame_pos).view(-1, max_frame_seq_len)
    frame_sen_pos = torch.LongTensor(pad_f_sen_pos).view(-1, max_frame_seq_len)
    return frame, frame_pos, frame_sen_pos, targets, targets_pos, targets_sen_pos


def get_ROC_loader(text, roc_vocab, frame_vocab, batch_size, shuffle, num_workers, fixed_len=False, is_flat = False):
    ROC = ROCDataset(roc_vocab,
                     frame_vocab,
                     text_path=text)

    data_loader = torch.utils.data.DataLoader(dataset=ROC,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ROC_collate_fn)
    return data_loader

def get_VIST_loader(VIST, roc_vocab, frame_vocab, batch_size, shuffle, num_workers):

    data_loader = torch.utils.data.DataLoader(dataset=VIST,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ROC_collate_fn)
    return data_loader

def get_window_loader(VIST, roc_vocab, frame_vocab, batch_size, shuffle, num_workers):

    data_loader = torch.utils.data.DataLoader(dataset=VIST,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ROC_added_termset_collate_fn)
    return data_loader
