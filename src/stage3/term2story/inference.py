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

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.cuda = False
    # Prepare DataLoader

    Dataloader = Loaders()
    Dataloader.get_test_loaders(opt)
    test_loader = Dataloader.loader['vist_term']

    opt.src_vocab_size = len(Dataloader.frame_vocab)
    opt.tgt_vocab_size = len(Dataloader.story_vocab)

    output = json.load(open('../data/generated_terms/VIST_test_self_output_diverse.json'))
    #output = json.load(open('../../commen-sense-storytelling/data/remove_bus_test.json'))
    count=0
    BOS_set = set([2,3,4,5,6,7])
    translator = Translator(opt)

    with open("./test/"+opt.output, 'w', buffering=1) as f_pred,\
            open("./test/"+'gt.txt', 'w', buffering=1) as f_gt, open("./test/"+'show.txt', 'w', buffering=1) as f:
        for frame, frame_pos, frame_sen_pos, gt_seqs, _, _ in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores = translator.translate_batch(frame, frame_pos, frame_sen_pos)
            for idx_seqs, idx_frame, gt_seq in zip(all_hyp, frame, gt_seqs):
                for idx_seq in idx_seqs:

                    f.write('Prediction:' + '\n')
                    pred_line = ' '.join([Dataloader.story_vocab.idx2word[idx] for idx in idx_seq if idx not in BOS_set])
                    f.write((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
                    f.write('Frame:' + '\n')
                    pred_line = ' '.join([Dataloader.frame_vocab.idx2word[idx.item()] for idx in idx_frame if idx!=Constants.PAD])
                    f.write((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
                    f.write('Ground Truth:' + '\n')
                    pred_line = ' '.join([Dataloader.story_vocab.idx2word[idx.item()] for idx in gt_seq if idx!=Constants.PAD])
                    f.write((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
                    f.write("===============================================\n")

                    pred_line = ' '.join([Dataloader.story_vocab.idx2word[idx] for idx in idx_seq if idx!=Constants.PAD])
                    f_pred.write((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
                    pred_line = ' '.join([Dataloader.story_vocab.idx2word[idx.item()] for idx in gt_seq if idx!=Constants.PAD])
                    f_gt.write((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
                    for _ in range(5):
                        output[count]['predicted_story']=pred_line = ' '.join(
                            [Dataloader.story_vocab.idx2word[idx] for idx in idx_seq if idx not in BOS_set])
                        count+=1

    print('[Info] Finished.')
    filename = '../data/generated_story/VIST_test_self_output_diverse_noun2_norm_penalty_VISTData_percent.json' + str(opt.positional) + '.json'
    
    json.dump(output, open(filename,'w'), indent=4)
if __name__ == "__main__":
    main()
