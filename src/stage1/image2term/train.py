'''
This script handling the training process.
'''

# Import comet_ml in the top of your file
#from comet_ml import Experiment
# Create an experiment
#experiment = Experiment(api_key="CUJif0Qw9Dr1IB74btNCwheMS",
#                                project_name="general", workspace="joe32140")
import argparse
import math
import time
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import model.Constants as Constants
from model.Models import ImageTransformer
from model.Optim import ScheduledOptim
#import adabound
import DataLoader
from build_story_vocab import Vocabulary


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing, tgt_vocab, opt):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    count = 0
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        image_ids, src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos, paths, path_order \
                                                                            = map(lambda x: x.to(device), batch)
        B, img_num, seq_len = tgt_seq.size()
        tgt_seq = tgt_seq.view(B*img_num, seq_len)
        tgt_pos = tgt_pos.view(B*img_num, seq_len)
        tgt_sen_pos = tgt_sen_pos.view(B*img_num, seq_len)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos, paths, path_order)
        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()
        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

        count +=1
        if count%100==0:
            print('Prediction:' + '\n')
            pred_line = ' '.join([tgt_vocab.idx2word[idx.item()] for idx in torch.topk(pred[0:opt.dec_max_seq_len],1,1)[1]])
            print((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
            print('Ground Truth:' + '\n')
            pred_line = ' '.join([tgt_vocab.idx2word[idx.item()] for idx in tgt_seq[0][:opt.dec_max_seq_len]])
            print((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
            print(f"Loss: {loss.item()/n_word}")
            print("===============================================\n")
        #experiment.log_metric("train_loss", loss.item()/n_word)

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device, opt, tgt_vocab):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            image_ids, src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos, paths, path_order \
                                                                                            = map(lambda x: x.to(device), batch)
            B, img_num, seq_len = tgt_seq.size()
            tgt_seq = tgt_seq.view(B*img_num, seq_len)
            tgt_pos = tgt_pos.view(B*img_num, seq_len)
            tgt_sen_pos = tgt_sen_pos.view(B*img_num, seq_len)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos, paths, path_order)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

            # test
            print('Prediction:' + '\n')
            pred_line = ' '.join([tgt_vocab.idx2word[idx.item()] for idx in torch.topk(pred[0:opt.dec_max_seq_len],1,1)[1]])
            print((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
            print('Ground Truth:' + '\n')
            pred_line = ' '.join([tgt_vocab.idx2word[idx.item()] for idx in tgt_seq[0][:opt.dec_max_seq_len]])
            print((pred_line + '\n').encode('ascii', 'ignore').decode('ascii'))
            print(f"Loss: {loss.item()/n_word}")
            print("===============================================\n")

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    #experiment.log_metric("eval_loss", loss_per_word)
    return loss_per_word, accuracy

def train(model, training_data, validation_data, optimizer, device, opt, tgt_vocab):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None
    log_dir = "./log/vist_"+opt.log
    if opt.model != None: log_dir = log_dir + "_pretrain"

    for param_group in optimizer._optimizer.param_groups:
        lr = param_group['lr']
        print(f"Scheduled Learning Rate:{lr}")
    if opt.log:
        log_train_file = log_dir + '.train.log'
        log_valid_file = log_dir + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing, tgt_vocab=tgt_vocab, opt=opt)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt, tgt_vocab)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            save_dir = "./save_vist_model_"+opt.log
            if opt.model != None: save_dir = save_dir + "_pretrain"
            save_dir = save_dir + "/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if opt.save_mode == 'all':
                model_name = save_dir + opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = save_dir+opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    #parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=16)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    #parser.add_argument('-d_model', type=int, default=512)
    #parser.add_argument('-d_inner_hid', type=int, default=2048)
    #parser.add_argument('-d_k', type=int, default=64)
    #parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=1)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', type=str, default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-device', type=str, default='0')
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-tgt_mode', default='term')
    parser.add_argument('-input_type', default='image')
    parser.add_argument('-tgt_type', default='term')
    parser.add_argument('-self_att', action='store_true', default=False)
    parser.add_argument('-global_att', action='store_true', default=False)
    parser.add_argument('-concept_path', action='store_true', default=False)
    parser.add_argument('-decoder', type=str, choices=['rnn', 'transformer'], default="transformer")


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    #data = torch.load(opt.data)
    #opt.max_token_seq_len = data['settings'].max_token_seq_len
    opt.enc_max_seq_len = 125
    opt.dec_max_seq_len = 10 if opt.tgt_type == "term" else 25

    torch.manual_seed(1234)
    loaders, src_vocab, tgt_vocab, concept_vocab = DataLoader.get_my_loaders(opt)
    training_data, validation_data, test_data= loaders['train'], loaders['val'], loaders['test']

    opt.src_vocab_size = len(src_vocab)
    opt.tgt_vocab_size = len(tgt_vocab)
    opt.concept_vocab_size = len(concept_vocab)

    # Report any information you need by:
    #experiment.log_multiple_params(vars(opt))
    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device(f'cuda:{opt.device}' if opt.cuda else 'cpu')

    transformer = ImageTransformer(
        opt.tgt_vocab_size,
        opt.concept_vocab_size,
        opt.enc_max_seq_len,
        opt.dec_max_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        decoder=opt.decoder,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        global_att=opt.global_att,
        concept_path=opt.concept_path).to(device)

    if opt.model:
        checkpoint = torch.load(opt.model)
        transformer.load_state_dict(checkpoint['model'])
        n_step = len(training_data)*checkpoint['epoch']
        print(f"N steps {n_step}")
        print(f"Loaded Pretrained Model: {opt.model}")

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)
    #if opt.model: optimizer.n_current_steps = n_step

    train(transformer, training_data, validation_data, optimizer, device ,opt, tgt_vocab)


if __name__ == '__main__':
    main()
