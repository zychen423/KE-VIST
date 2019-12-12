''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import json
from transformer.Translator import Translator
from Loader_manager import Loaders
from build_story_vocab import Vocabulary
from transformer import Constants
import re

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=3,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-device', action='store')
    parser.add_argument('-positional', type = str, choices=['Default', 'LRPE','LDPE'], default = 'Default')
    parser.add_argument('-insert', required=True, type=int, help="Term's insert number")
    parser.add_argument('-relation', required=True, type=str, help="relation file")
    
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    print('opt.insert',opt.insert)
    print('opt.relation',opt.relation)
    Dataloader = Loaders()
    Dataloader.get_test_add_path_loaders(opt)
    test_loader = Dataloader.loader['add_window_termset']

    opt.src_vocab_size = len(Dataloader.frame_vocab)
    opt.tgt_vocab_size = len(Dataloader.story_vocab)

    output = json.load(open('../data/generated_terms/VIST_test_self_output_diverse.json'))
    count=0
    BOS_set = set([2,3,4,5,6,7])
    BOSs_re = "|".join([Dataloader.story_vocab.idx2word[idx] for idx in Constants.BOSs])
    translator = Translator(opt)

    with open("./test/"+opt.output, 'w', buffering=1) as f_pred,\
            open("./test/"+'gt.txt', 'w', buffering=1) as f_gt, open("./test/"+'show.txt', 'w', buffering=1) as f:
        for frame, frame_pos, frame_sen_pos, gt_seqs, _, _ in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores = translator.translate_batch(frame, frame_pos, frame_sen_pos)
            print('frame',frame)
            print('frame_sen_pos',frame_sen_pos)
            print('all_hyp',all_hyp)
            window_stories=[]
            for idx_seqs, idx_frame, gt_seq in zip(all_hyp, frame, gt_seqs):
                for idx_seq in idx_seqs:
                    pred_line = ' '.join(
                            [Dataloader.story_vocab.idx2word[idx] for idx in idx_seq if idx != Constants.EOS])
                    window_stories.append(re.split( BOSs_re, pred_line))
            tmp_added_story = window_stories[0]
            if len(window_stories)>1:
                for i in range(1, len(window_stories)):
                    tmp_added_story.append(window_stories[i][-1])
            tmp_added_story = " ".join(tmp_added_story)
            for i in range(count*5, count*5+5):
                output[i]['add_one_path_story'] = re.sub(' +', ' ', tmp_added_story)
            count+=1
    print('[Info] Finished.')
    
    filename = '../data/generated_story/VIST_test_self_output_diverse_add_highest_one_path_noun' + str(opt.insert+1) + str(opt.relation) + '_norm_penalty_coor_VISTdataset_percent_' + str(opt.positional) + '.json'
    
    json.dump(output, open(filename,'w'), indent=4)
if __name__ == "__main__":
    main()
