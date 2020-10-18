import DataLoader
import pickle
from build_term_vocab import Vocabulary
import torch
class Loaders():

    def __init__(self):
        self.loader ={}
        with open("../data/term2story_vocabs/term_vocab.pkl",'rb') as f:
        #with open("../../event-visual-storytelling/data/ROC_Frame_vocab.pkl",'rb') as f:
            self.frame_vocab = pickle.load(f)
        with open("../data/term2story_vocabs/story_vocab.pkl",'rb') as f:
        #with open("../../event-visual-storytelling/data/ROC_Story_vocab.pkl",'rb') as f:
            self.story_vocab = pickle.load(f)
        print(self.story_vocab("."))

    def get_loaders(self, args):
        STORY_TERM_PATH = "../data/ROC/ROC_{}.json"

        self.loader['train'] = DataLoader.get_ROC_loader(STORY_TERM_PATH.format('train'),
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         True, 5)
        self.loader['val'] = DataLoader.get_ROC_loader(STORY_TERM_PATH.format('valid'),
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         True, 5)
        self.loader['test'] = DataLoader.get_ROC_loader(STORY_TERM_PATH.format('test'),
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 5)
        STORY_TERM_PATH = "../data/VIST/VIST_coref_nos_mapped_frame_noun_train.json"

        Dataset = DataLoader.VISTDataset(self.story_vocab,
                         self.frame_vocab,
                         text_path=STORY_TERM_PATH)
        self.loader['vist_train'] = DataLoader.get_VIST_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size//4+1,
                                                         True, 5)
        STORY_TERM_PATH = "../data/VIST/VIST_coref_nos_mapped_frame_noun_val.json"

        Dataset = DataLoader.VISTDataset(self.story_vocab,
                         self.frame_vocab,
                         text_path=STORY_TERM_PATH)
        self.loader['vist_val'] = DataLoader.get_VIST_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 5)
        
        
        
    def get_test_loaders(self, args):
        #add_termset = '../data/test_roc_add_term.json'

        add_termset = '../data/added_path_stories/language_model_terms_add_lowest.json'
        print(f"test data name:{add_termset}")
        Dataset = DataLoader.ROCAddTermsetDataset(self.story_vocab,
                         self.frame_vocab,
                         text_path=add_termset)
        self.loader['add_window_termset'] = DataLoader.get_window_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         1,
                                                         False, 1)

        STORY_TERM_PATH = "../data/generated_terms/VIST_test_self_output_diverse.json"
        Dataset = DataLoader.PredictedVISTDataset(self.story_vocab,
                         self.frame_vocab,
                         text_path=STORY_TERM_PATH)
        self.loader['vist_term'] = DataLoader.get_VIST_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 5)
    def get_test_add_path_loaders(self, args):
        #add_termset = '../data/test_roc_add_term.json'
        add_termset = args.term_path
        
        print(f"test data name:{add_termset}")
        Dataset = DataLoader.ROCAddTermsetDataset(self.story_vocab,
                         self.frame_vocab,
                         text_path=add_termset)
        self.loader['add_window_termset'] = DataLoader.get_window_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         1,
                                                         False, 1)

        STORY_TERM_PATH = "../data/generated_terms/VIST_test_self_output_diverse.json"
        Dataset = DataLoader.PredictedVISTDataset(self.story_vocab,
                         self.frame_vocab,
                         text_path=STORY_TERM_PATH)
        self.loader['vist_term'] = DataLoader.get_VIST_loader(Dataset,
                                                         self.story_vocab,
                                                         self.frame_vocab,
                                                         args.batch_size,
                                                         False, 5)

