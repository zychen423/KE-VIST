''' Translate input text with trained model. '''
# -*- coding: utf-8 -*-
import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import json
from model.Translator import Translator
import DataLoader
from build_story_vocab import Vocabulary
from model import Constants
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
    parser.add_argument('-input_type', default="image")
    parser.add_argument('-tgt_type', default='term')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.dec_max_seq_len = 10 if opt.tgt_type == "term" else 25
    # Prepare DataLoader
    loaders, src_vocab, tgt_vocab, concept_vocab = DataLoader.get_my_loaders(opt)
    training_data, validation_data, test_loader= loaders['train'], loaders['val'], loaders['test']

    opt.src_vocab_size = len(src_vocab)
    opt.tgt_vocab_size = len(tgt_vocab)
    opt.concept_vocab_size = len(concept_vocab)

    translator = Translator(opt)
    count=0
    output = json.load(open('../data/VIST/VIST_train_noun.json'))
    stop_set = set([Constants.PAD, Constants.EOS])
    pre_set = set([Constants.PAD, Constants.EOS, tgt_vocab('.')])

    with open("./test/"+opt.output, 'w', buffering=1) as f_pred, open("./test/"+'gt.txt', 'w', buffering=1) as f_gt, open("./test/"+'show.txt', 'w', buffering=1) as f:
        for image_ids, frame, frame_pos, frame_sen_pos, gt_seqs, _, _, paths, path_pos in tqdm(training_data, mininterval=2, desc='  - (Test)', leave=False):
            # prepare data
            B, img_num, seq_len = gt_seqs.size()
            B, img_num, obj_num, term_len = frame.size()
            B, img_num, path_num, paths_len= paths.size()
            pre_words=[]
            for k in range(5):
                image_ids_, frame_, frame_pos_, frame_sen_pos_, gt_seqs_, _, _, paths_, path_pos_=\
                        image_ids[:,k].view(B,1), frame[:,k].view(B,1,obj_num,term_len), frame_pos[:,k].view(B,1,obj_num), frame_sen_pos[:,k].view(B,1,obj_num), gt_seqs[:,k].view(1,seq_len), _, _, paths[:,k].view(B,1,path_num, paths_len), path_pos[:,k].view(B,1,path_num)
                all_hyp, all_scores = translator.translate_batch(frame_, frame_pos_, frame_sen_pos_, paths_, path_pos_, pre_words)
                for idx_seqs, gt_seq, img, img_id in zip(all_hyp, gt_seqs_, frame_, image_ids_[0]):
                    for idx_frame, idx_seq in zip(frame, idx_seqs):
                        f.write(f'Image ID: {img_id}\n')
                        if opt.input_type != 'image':
                            f.write('Object Action:' + '\n')
                            for obj in img:
                                pred_line = ' '.join([src_vocab.idx2word[idx.item()] for idx in obj if idx!=Constants.PAD])
                                f.write(pred_line + '\n')
                        f.write('Prediction:' + '\n')
                        pred_line = ' '.join([tgt_vocab.idx2word[idx] for idx in idx_seq if idx not in stop_set])
                        f.write((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
                        f.write('Ground Truth:' + '\n')
                        pred_line = ' '.join([tgt_vocab.idx2word[idx.item()] for idx in gt_seq if idx!=Constants.PAD])
                        f.write((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
                        f.write("===============================================\n")

                        output[count]['predicted_term_seq']=[src_vocab.idx2word[idx] for idx in idx_seq if idx not in stop_set ]
                        count+=1
                        for rm_id in idx_seq:
                            if rm_id not in pre_set:
                                pre_words.append(rm_id)

            f.write("-----------Next Story-------------\n")

    print('[Info] Finished.')
    json.dump(output, open(f'../data/generated_terms/VIST_text_self_output_diverse_train.json', 'w'), indent=4)
if __name__ == "__main__":
    main()
