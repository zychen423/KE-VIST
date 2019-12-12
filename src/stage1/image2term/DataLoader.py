import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import os
import pickle
import nltk
import json
from build_story_vocab import Vocabulary
from model import Constants
import numpy as np


class VisualStorytellingDataset(Dataset):
    def __init__(self,
                root='/home/joe32140/data/VIST/images/all_images/',
                resnet_feat_path = '/home/troutman/projects/AREL/DATADIR/resnet_features/fc/',
                text_path='../data/sis/modified_val.story-in-sequence.json',
                tgt_vocab=None,
                transform=None):
        # For image processing
        self.root = root
        self.resnet_feat_path = resnet_feat_path
        self.transform = transform

        # Text generation
        self.dialogs = json.load(open(text_path, 'r'))
        self.max_tgt_len = 24
        self.tgt_vocab = tgt_vocab

    def __getitem__(self, index):

        targets = []
        image_feats = []
        image_ids = []
        for idx in range(index*5,index*5+5):
            dialog = self.dialogs[idx]
            tmp_tgt = []
            image_id = dialog['photo_flickr_id']
            image_feat = np.load(os.path.join(self.resnet_feat_path, image_id + '.npy'))
            image_feats.append(image_feat)

            tgt_text = dialog['text'].lower()
            tokens = nltk.word_tokenize(tgt_text)
            tmp_tgt.extend([self.tgt_vocab(token) for token in tokens])
            if len(tmp_tgt) > self.max_tgt_len-1:
                tmp_tgt = tmp_tgt[:self.max_tgt_len]

            targets.append(tmp_tgt)
            image_ids.append(int(image_id))

        T, T_sen_pos, T_word_pos, I, I_sen_pos, I_word_pos = [], [], [], [], [], []
        for i, t in enumerate(targets):
            T.append(Constants.BOSs[i])
            T.extend(t)
            T_sen_pos.extend([i+1]*(len(t)+1))
            T_word_pos.extend([i+1 for i in range(len(t)+1)])

        assert len(T) == len(T_sen_pos)
        assert len(T) == len(T_word_pos)

        I = image_feats
        I_sen_pos = range(1,6)
        I_word_pos = range(1,6)

        return T, T_sen_pos, T_word_pos, I,I_sen_pos, I_word_pos, image_ids

    def __len__(self):
        return len(self.dialogs)//5

    def collate_fn(self, data):

        #List of sentences and frames [B,]
        tgt_text, tgt_sen_pos, tgt_word_pos, image_feats, img_sen_pos, img_word_pos, image_ids = zip(*data)

        lengths = [len(x)+1 for x in tgt_text]
        #max_seq_len = max(lengths)
        max_tgt_len = 126
        padded_tgt_text = [t+ [Constants.EOS] + [Constants.PAD for _ in range(max_tgt_len - len(t) - 1)] for t in tgt_text]
        #stories_pos = [[pos_i+1 if w_i != 0 else 0
        #     for pos_i, w_i in enumerate(inst)] for inst in pad_stories]
        padded_tgt_sen_pos = [t + [Constants.PAD for _ in range(max_tgt_len - len(t))] for t in tgt_sen_pos]
        padded_tgt_word_pos = [t + [Constants.PAD for _ in range(max_tgt_len - len(t))] for t in tgt_word_pos]


        img_lengths = [len(x) for x in image_feats]
        #max_frame_seq_len = max(frame_lengths)
        max_img_seq_len = 126

        targets = torch.LongTensor(padded_tgt_text).view(-1, max_tgt_len)
        lengths = torch.LongTensor(lengths).view(-1,1)
        targets_pos = torch.LongTensor(padded_tgt_word_pos).view(-1, max_tgt_len)
        targets_sen_pos = torch.LongTensor(padded_tgt_sen_pos).view(-1, max_img_seq_len)
        image_feats = torch.FloatTensor(image_feats).view(-1, 5, 2048)
        img_lengths = torch.LongTensor(img_lengths).view(-1,1)
        img_word_pos = torch.LongTensor(img_word_pos).view(-1, 5)
        img_sen_pos = torch.LongTensor(img_sen_pos).view(-1, 5)
        image_ids = torch.tensor(image_ids)
        return image_ids, image_feats, img_word_pos, img_sen_pos, targets, targets_pos, targets_sen_pos

class ObjectVisualStorytellingDataset(Dataset):
    def __init__(self,
                text_path='../data/sis/modified_val.story-in-sequence.json',
                objects_dic=None,
                src_vocab=None,
                tgt_vocab=None):

        # Text generation
        self.dialogs = json.load(open(text_path, 'r'))
        self.object_action_pairs = objects_dic
        self.max_tgt_len = 5
        self.max_obj_len = 10
        self.max_term_len = 4
        self.src_voacb = src_vocab
        self.tgt_vocab = tgt_vocab

    def __getitem__(self, index):

        targets = []
        obj_act_pairs = []
        image_ids = []
        for idx in range(index*5,index*5+5):
            dialog = self.dialogs[idx]
            tmp_tgt = []
            image_id = dialog['photo_flickr_id']
            try:
                object_action = self.object_action_pairs[str(image_id)+'.jpg']['object_action'] # list of list of terms
            except:
                object_action = self.object_action_pairs[str(image_id)+'.png']['object_action'] # list of list of terms
            obj_act = [[self.src_voacb(terms[i]) for i in range(len(terms)) if i < self.max_term_len] for terms in object_action]
            if len(obj_act) > self.max_obj_len:
                obj_act = obj_act[:self.max_obj_len]

            #tgt_text = dialog['text'].lower()
            #tokens = nltk.word_tokenize(tgt_text)
            tokens = dialog['text_mapped_with_nouns_and_frame']
            tmp_tgt.extend([self.tgt_vocab(token) for token in tokens])
            if len(tmp_tgt) > self.max_tgt_len:
                tmp_tgt = tmp_tgt[:self.max_tgt_len]

            targets.append(tmp_tgt)
            obj_act_pairs.append(obj_act)
            image_ids.append(int(image_id))
        return targets, obj_act_pairs, image_ids # [5, 24], [5, 10, 4], [5]

    def __len__(self):
        return len(self.dialogs)//5

    def collate_fn(self, data):
        tgt_text, obj_act, image_ids = zip(*data) # [B, 5, ~24], [B, 5, ~10, ~4], [B, 5]

        src_lengths = [[len(x)+1 for x in img] for img in obj_act]
        padded_src =[ [[t + [Constants.PAD for _ in range(self.max_term_len - len(t))] for t in objs]
                                                                    for objs in img] for img in obj_act]
        for i, img in enumerate(padded_src):
            for j, objs in enumerate(img):
                if len(objs) < self.max_obj_len:
                    padded_src[i][j].extend([[Constants.PAD]*self.max_term_len for i in range(self.max_obj_len-len(objs))])
        padded_src_order =[[[order+1 if k<src_lengths[i][order] else 0 for k, obj in enumerate(objs) ]
                        for order, objs in enumerate(img)] for i, img in enumerate(padded_src)]
        #lengths = [[len(x)+1 for x in img] for img in tgt_text]
        #max_seq_len = max(lengths)
        max_tgt_len = 7
        padded_tgt_text = [[[Constants.BOS]+t+[Constants.EOS] + [Constants.PAD for _ in range(max_tgt_len - len(t)-2)] for t in img]   for img in tgt_text]
        #stories_pos = [[pos_i+1 if w_i != 0 else 0
        #     for pos_i, w_i in enumerate(inst)] for inst in pad_stories]
        padded_tgt_sen_pos =[[[order+1 if token != 0 else 0 for token in sentence]
                                                            for order, sentence in enumerate(img)] for img in padded_tgt_text]
        padded_tgt_word_pos =[[[pos+1 if token != 0 else 0 for pos, token in enumerate(sentence)]
                                                         for sentence in img] for img in padded_tgt_text]
        #max_frame_seq_len = max(frame_lengths)

        targets = torch.LongTensor(padded_tgt_text).view(-1, 5, max_tgt_len)
        #tgt_lengths = torch.LongTensor(lengths).view(-1, 5, 1)
        targets_pos = torch.LongTensor(padded_tgt_word_pos).view(-1, 5, max_tgt_len)
        targets_sen_pos = torch.LongTensor(padded_tgt_sen_pos).view(-1, 5, max_tgt_len)

        src = torch.LongTensor(padded_src).view(-1, 5, self.max_obj_len, self.max_term_len)
        src_lengths = torch.LongTensor(src_lengths).view(-1, 5, 1)
        #src_word_pos = torch.LongTensor(padded_src_order).view(-1, 5, self.max_obj_len)
        src_order_pos = torch.LongTensor(padded_src_order).view(-1, 5, self.max_obj_len)
        image_ids = torch.tensor(image_ids)
        return image_ids, src, src_order_pos, src_order_pos, targets, targets_pos, targets_sen_pos

class ObjectImageFeatureVisualStorytellingDataset(Dataset):
    def __init__(self,
                text_path='../data/sis/modified_val.story-in-sequence.json',
                image_feat_path='../bottom-up-attention/vist_data/vist_att',
                conceptnet_path='./data/storyid2concept_tree_objection_detection_train.json',
                tgt_vocab=None,
                concept_vocab=None,
                tgt_type='term',
                max_tgt_len=10):

        # Text generation
        self.dialogs = json.load(open(text_path, 'r'))
        self.image_feat_path = image_feat_path
        self.conceptnet = json.load(open(conceptnet_path, 'r'))
        self.max_path_num =10
        self.max_hop = 5
        self.max_tgt_len = max_tgt_len-2
        self.max_obj_len = 25
        self.tgt_vocab = tgt_vocab
        self.concept_vocab = concept_vocab
        self.tgt_type = tgt_type

    def __getitem__(self, index):

        targets = []
        image_obj_feats = []
        image_ids = []
        for idx in range(index*5,index*5+5):
            dialog = self.dialogs[idx]
            story_id = dialog['story_id']
            tmp_tgt = []
            image_id = dialog['photo_flickr_id']
            obj_feats = np.load(self.image_feat_path+image_id+'.npz')['feat']
            if len(obj_feats) > self.max_obj_len:
                obj_feats = obj_feats[:self.max_obj_len]

            if self.tgt_type == 'text':
                tgt_text = dialog['text'].lower()
                tokens = nltk.word_tokenize(tgt_text)
            else:
                tokens = dialog['coref_mapped_seq']
            tmp_tgt.extend([self.tgt_vocab(token) for token in tokens])
            if len(tmp_tgt) > self.max_tgt_len:
                tmp_tgt = tmp_tgt[:self.max_tgt_len]

            targets.append(tmp_tgt)
            image_obj_feats.append(obj_feats)
            image_ids.append(int(image_id))

        # Common Sense Extraction
        image_paths = self.conceptnet[story_id]
        relations = [[[self.concept_vocab(entity) for entity in path]
                                 for i ,path in enumerate(paths) if i<self.max_path_num ] for paths in image_paths]

        return targets, image_obj_feats, relations, image_ids # [5, 24], [5, 10, 2048], [4, ~10, 3~5], [5]

    def __len__(self):
        return len(self.dialogs)//5

    def collate_fn(self, data):
        tgt_text, image_obj_feats, conceptnet_relations, image_ids = zip(*data) # [B, 5, ~24], [B, 5, ~10, 2048], [B, 4, ~10, ~5], [B, 5]

        # -- Pad image features
        src_lengths = [[len(x)+1 for x in img] for img in image_obj_feats]
        for i, img in enumerate(image_obj_feats):
            for j, objs in enumerate(img):
                if len(objs) < self.max_obj_len:
                    image_obj_feats[i][j] = np.concatenate((image_obj_feats[i][j],
                                                np.zeros((self.max_obj_len-len(objs), 2048))), axis=0)
        padded_src_order =[[[order+1 if k<src_lengths[i][order] else 0 for k, obj in enumerate(objs) ]
                        for order, objs in enumerate(img)] for i, img in enumerate(image_obj_feats)]

        # -- Pad concept paths
        relation_lengths = [[len(x)+1 for x in img] for img in conceptnet_relations]
        padded_relations =[ [[t + [Constants.PAD for _ in range(self.max_hop - len(t))] for t in paths]
                                                                    for paths in img] for img in conceptnet_relations]
        for i, img in enumerate(padded_relations):
            for j, paths in enumerate(img):
                if len(paths) < self.max_path_num:
                    padded_relations[i][j].extend([[Constants.PAD]*self.max_hop for i in range(self.max_path_num-len(paths))])

        for i, _ in enumerate(padded_relations):
            padded_relations[i].append([[Constants.PAD]*self.max_hop for _ in range(self.max_path_num)])

        padded_relation_order =[[[order+1 if order<4 and k<relation_lengths[i][order] else 0 for k, path in enumerate(paths) ]
                        for order, paths in enumerate(img)] for i, img in enumerate(padded_relations)]


        # -- Pad tgt story
        max_tgt_len = self.max_tgt_len+2
        padded_tgt_text = [[[Constants.BOS]+t+[Constants.EOS] + [Constants.PAD for _ in range(max_tgt_len - len(t)-2)] for t in img]   for img in tgt_text]
        #stories_pos = [[pos_i+1 if w_i != 0 else 0
        #     for pos_i, w_i in enumerate(inst)] for inst in pad_stories]
        padded_tgt_sen_pos =[[[order+1 if token != 0 else 0 for token in sentence]
                                                            for order, sentence in enumerate(img)] for img in padded_tgt_text]
        padded_tgt_word_pos =[[[pos+1 if token != 0 else 0 for pos, token in enumerate(sentence)]
                                                         for sentence in img] for img in padded_tgt_text]
        targets = torch.LongTensor(padded_tgt_text).view(-1, 5, max_tgt_len)
        #tgt_lengths = torch.LongTensor(lengths).view(-1, 5, 1)
        targets_pos = torch.LongTensor(padded_tgt_word_pos).view(-1, 5, max_tgt_len)
        targets_sen_pos = torch.LongTensor(padded_tgt_sen_pos).view(-1, 5, max_tgt_len)

        src = torch.FloatTensor(image_obj_feats).view(-1, 5, self.max_obj_len, 2048)
        src_lengths = torch.LongTensor(src_lengths).view(-1, 5, 1)
        #src_word_pos = torch.LongTensor(padded_src_order).view(-1, 5, self.max_obj_len)
        src_order_pos = torch.LongTensor(padded_src_order).view(-1, 5, self.max_obj_len)
        relation = torch.LongTensor(padded_relations).view(-1, 5, self.max_path_num, self.max_hop)
        relation_lengths = torch.LongTensor(relation_lengths).view(-1, 4, 1)
        relation_order_pos = torch.LongTensor(padded_relation_order).view(-1, 5, self.max_path_num)
        image_ids = torch.tensor(image_ids)
        return image_ids, src, src_order_pos, src_order_pos, targets, targets_pos, targets_sen_pos, relation, relation_order_pos

def get_loader(root, feat_path, text, vocab, batch_size, shuffle, num_workers):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])
    VS = VisualStorytellingDataset(root=root,
                       resnet_feat_path=feat_path,
                       text_path=text,
                       tgt_vocab=vocab,
                       transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=VS,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=VS.collate_fn)
    return data_loader


def get_loaders(args):

    IMG_PATH = '/home/joe32140/data/VIST/images/all_images/'
    IMG_FEAT_PATH = '/home/troutman/projects/AREL/DATADIR/resnet_features/fc/{}/'
    DATA_PATH = '/home/joe32140/image2story/data/VIST_{}_noun.json'
    with open(f"./data/{args.tgt_mode}_vocab.pkl",'rb') as f:
        vocab = pickle.load(f)

    loader={}
    loader['train'] = get_loader(IMG_PATH,
                                IMG_FEAT_PATH.format('train'),
                                DATA_PATH.format('train'),
                                vocab,
                                args.batch_size, True, 5)
    loader['val'] = get_loader(IMG_PATH,
                                IMG_FEAT_PATH.format('val'),
                                DATA_PATH.format('val'),
                                vocab,
                                args.batch_size, True, 5)
    loader['test'] = get_loader(IMG_PATH,
                                IMG_FEAT_PATH.format('test'),
                                DATA_PATH.format('test'),
                                vocab,
                                args.batch_size, False, 5)
    return loader, vocab

def get_obj_loader(text, objects_dic, src_vocab, tgt_vocab, batch_size, shuffle, num_workers):
    OVS = ObjectVisualStorytellingDataset(
                       text_path=text,
                       objects_dic=objects_dic,
                       src_vocab=src_vocab,
                       tgt_vocab=tgt_vocab)

    data_loader = torch.utils.data.DataLoader(dataset=OVS,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=OVS.collate_fn)
    return data_loader


def get_obj_loaders(args):

    DATA_PATH = './data/VIST_{}_with_noun_and_frame.json'
    Object_Term_Path='/home/swat230/open-sesame/Vist_image_captioning_object_action_pair.json'
    with open(f"./data/frame_vocab.pkl",'rb') as f:
        src_vocab = pickle.load(f)
    with open(f"./data/{args.tgt_type}_vocab.pkl",'rb') as f:
        tgt_vocab = pickle.load(f)
    Objects = json.load(open(Object_Term_Path))
    loader={}
    loader['train'] = get_obj_loader(DATA_PATH.format('train'),
                                Objects,
                                src_vocab,
                                tgt_vocab,
                                args.batch_size, True, 5)
    loader['val'] = get_obj_loader(DATA_PATH.format('val'),
                                Objects,
                                src_vocab,
                                tgt_vocab,
                                args.batch_size, True, 5)
    loader['test'] = get_obj_loader(DATA_PATH.format('test'),
                                Objects,
                                src_vocab,
                                tgt_vocab,
                                args.batch_size, False, 5)
    return loader, src_vocab, tgt_vocab

def get_obj_feats_loader(text, image_feat_path, conceptnet_path, tgt_vocab, concept_vocab, batch_size, shuffle, num_workers, args):

    OVS = ObjectImageFeatureVisualStorytellingDataset(
                       text_path=text,
                       image_feat_path=image_feat_path,
                       conceptnet_path=conceptnet_path,
                       tgt_vocab=tgt_vocab,
                       concept_vocab=concept_vocab,
                       tgt_type=args.tgt_type,
                       max_tgt_len=args.dec_max_seq_len)

    data_loader = torch.utils.data.DataLoader(dataset=OVS,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=OVS.collate_fn)
    return data_loader

def get_obj_feats_loaders(args):

    #DATA_PATH = './data/VIST_coref_{}_mapped_frame_noun.json'
    DATA_PATH = '../data/VIST/VIST_coref_nos_mapped_frame_noun_{}.json'
    #image_feat_path='/home/joe32140/bottom-up-attention/vist_data/vist_att/'
    image_feat_path='/home/joe32140/bottom-up-attention/vist_data_fix_rotation/vist_att/'
    conceptnet_path='../data/concept/storyid2concept_tree_noun_and_frame_{}.json'
    #conceptnet_path='./data/storyid2concept_tree_{}.json'
    with open(f"../data/image2term_vocabs/{args.tgt_type}_vocab.pkl",'rb') as f:
        tgt_vocab = pickle.load(f)
    with open(f"../data/image2term_vocabs/concept_vocab.pkl",'rb') as f:
        concept_vocab = pickle.load(f)
    loader={}
    loader['train'] = get_obj_feats_loader(DATA_PATH.format('train'),
                                image_feat_path,
                                conceptnet_path.format('train'),
                                tgt_vocab,
                                concept_vocab,
                                args.batch_size, True, 10, args)

    loader['val'] = get_obj_feats_loader(DATA_PATH.format('val'),
                                image_feat_path,
                                conceptnet_path.format('val'),
                                tgt_vocab,
                                concept_vocab,
                                args.batch_size, True, 10, args)

    loader['test'] = get_obj_feats_loader(DATA_PATH.format('test'),
                                image_feat_path,
                                conceptnet_path.format('test'),
                                tgt_vocab,
                                concept_vocab,
                                args.batch_size, False, 10, args)
    return loader, tgt_vocab, tgt_vocab, concept_vocab

def get_my_loaders(args):
    if args.input_type == 'image':
        return get_obj_feats_loaders(args)
    else :
        return get_obj_loaders(args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    #parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-tgt_type', type=str, default='term')
    parser.add_argument('-input_type', type=str, default='image')
    opt = parser.parse_args()
    opt.dec_max_seq_len = 10
    dataset = get_obj_feats_loaders(opt)
    image_ids, src, src_order_pos, src_order_pos, targets, targets_pos, targets_sen_pos, relation, relation_lengths =  iter(dataset[0]['test']).next()
    print(src_order_pos)
