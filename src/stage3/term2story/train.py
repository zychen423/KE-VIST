'''
This script handling the training process.
'''

import argparse
import math
import time
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
#from revised_loader import revised_Loaders
from Loader_manager import Loaders
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
    #print('cal_loss, pred', pred.shape)
    #print('cal_loss, gold', gold.shape)
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


def train_epoch(model, training_data, optimizer, device, smoothing, Dataloader, opt):
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
        src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]
       # print('src_seq',src_seq)
        #print('src_seq[0]',src_seq[0])
        #print('src_seq.shape',src_seq.shape)
        #print('tgt_seq',tgt_seq)
        #print('tgt_seq.shape',tgt_seq.shape)

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos)
        #print('pred', pred[0:10] )
        #print('pred index')
        #print(' '.join([str(idx.item()) for idx in torch.topk(pred,1,1)[1]]))
        #print('pred[0][0:20]', pred[0][0:20])
        #print('pred[0].shape', pred[0].shape)
        #print('pred_shape', pred.shape)
        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        #print('loss',loss)
        #print('loss.item()',loss.item())
        #print('n_correct', n_correct)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()
        
        total_loss += loss.item()
        #print('total_loss', total_loss)
        non_pad_mask = gold.ne(Constants.PAD)
        #print('non_pad_mask', non_pad_mask)
        n_word = non_pad_mask.sum().item()
        #print('n_word', n_word)
        n_word_total += n_word
        n_word_correct += n_correct
        # note keeping
        count +=1
        if count%1500==0:
            try:
                print('Prediction:' + '\n')
                pred_line = ' '.join([Dataloader.story_vocab.idx2word[idx.item()] for idx in torch.topk(pred[0:60],1,1)[1]])
                print(pred_line + '\n')
                print('Frame:' + '\n')
                pred_line = ' '.join([Dataloader.frame_vocab.idx2word[idx.item()] for idx in src_seq[0][:60]])
                print(pred_line + '\n')
                print('Ground Truth:' + '\n')
                pred_line = ' '.join([Dataloader.story_vocab.idx2word[idx.item()] for idx in tgt_seq[0][:60]])
                print(pred_line + '\n')
                print(f"Loss: {loss.item()/non_pad_mask.sum().item()}")
                print("===============================================\n")
            except:
                pass

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train_vist_epoch(model, training_data, vist_train_data, optimizer, device, smoothing, Dataloader, opt):
    ''' Epoch operation in training phase'''

    model.train()
    roc_iter = iter(training_data)
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    count = 0
    for batch in tqdm(
            vist_train_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare vist data
        src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # Adding ROC data
        try:
            roc_batch = next(roc_iter)
        except StopIteration:
            roc_iter = iter(training_data)
            roc_batch = next(roc_iter)

        src_seq_c, src_pos_c, src_sen_pos_c, tgt_seq_c, tgt_pos_c, tgt_sen_pos_c = map(lambda x: x.to(device), roc_batch)
        gold_c = tgt_seq_c[:, 1:]

        src_seq = torch.cat((src_seq,src_seq_c), 0)
        src_pos = torch.cat((src_pos,src_pos_c), 0)
        src_sen_pos = torch.cat((src_sen_pos,src_sen_pos_c), 0)
        tgt_seq = torch.cat((tgt_seq,tgt_seq_c), 0)
        tgt_pos = torch.cat((tgt_pos,tgt_pos_c), 0)
        tgt_sen_pos = torch.cat((tgt_sen_pos,tgt_sen_pos_c), 0)
        gold = torch.cat((gold,gold_c), 0)

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct
        # note keeping
        count +=1
        if count%300==0:
            try:
                print('Prediction:' + '\n')
                pred_line = ' '.join([Dataloader.story_vocab.idx2word[idx.item()] for idx in torch.topk(pred[0:60],1,1)[1]])
                print(pred_line + '\n')
                print('Frame:' + '\n')
                pred_line = ' '.join([Dataloader.frame_vocab.idx2word[idx.item()] for idx in src_seq[0][:60]])
                print(pred_line + '\n')
                print('Ground Truth:' + '\n')
                pred_line = ' '.join([Dataloader.story_vocab.idx2word[idx.item()] for idx in tgt_seq[0][:60]])
                print(pred_line + '\n')
                print(f"Loss: {loss.item()/non_pad_mask.sum().item()}")
                print("===============================================\n")
            except:
                pass

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device, Dataloader, opt):
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
            src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, src_sen_pos, tgt_seq, tgt_pos, tgt_sen_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)
            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct
            # test
            try:
                print('Prediction:' + '\n')
                pred_line = ' '.join([Dataloader.story_vocab.idx2word[idx.item()] for idx in torch.topk(pred[0:60],1,1)[1]])
                print(pred_line + '\n')
                print('Frame:' + '\n')
                pred_line = ' '.join([Dataloader.frame_vocab.idx2word[idx.item()] for idx in src_seq[0][:60]])
                print(pred_line + '\n')
                print('Ground Truth:' + '\n')
                pred_line = ' '.join([Dataloader.story_vocab.idx2word[idx.item()] for idx in tgt_seq[0][:60]])
                print(pred_line + '\n')
                print(f"Loss: {loss.item()/non_pad_mask.sum().item()}")
                print("===============================================\n")
            except:
                pass


    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, validation_data, vist_train_data, vist_val_data, optimizer, device, opt, Dataloader):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None
    log_dir = "./log/roc"
    if opt.model != None: log_dir = log_dir + "_pretrain"
    if opt.vist: log_dir = log_dir + "_vist"

    if opt.log:
        log_train_file = log_dir+ opt.log + '.train.log'
        log_valid_file = log_dir+ opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        if opt.vist:
            train_loss, train_accu = train_vist_epoch(
                model, training_data, vist_train_data, optimizer, device, smoothing=opt.label_smoothing, Dataloader=Dataloader, opt=opt)
        else:
            train_loss, train_accu = train_epoch(
                model, training_data, optimizer, device, smoothing=opt.label_smoothing, Dataloader=Dataloader, opt=opt)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        if opt.vist:
            valid_loss, valid_accu = eval_epoch(model, vist_val_data, device, Dataloader, opt)
        else:
            valid_loss, valid_accu = eval_epoch(model, validation_data, device, Dataloader, opt)
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
            save_dir = "./save_model_" + opt.log + opt.positional
            if opt.model != None: save_dir = save_dir + "_pretrain"
            if opt.vist: save_dir = save_dir + "_vist"
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
    parser.add_argument('-batch_size', type=int, default=64)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    #parser.add_argument('-d_model', type=int, default=512)
    #parser.add_argument('-d_inner_hid', type=int, default=2048)
    #parser.add_argument('-d_k', type=int, default=64)
    #parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-n_warmup_steps', type=int, default=2000)
    #parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-device', type=str, default='0')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-positional', type = str, choices=['Default', 'LRPE','LDPE'], default = 'Default')

    parser.add_argument('-vist', default=False, action='store_true')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    #========= Loading Dataset =========#
    #data = torch.load(opt.data)
    #opt.max_token_seq_len = data['settings'].max_token_seq_len

    opt.max_encode_token_seq_len = 51
    opt.max_token_seq_len = 126

    torch.manual_seed(1234)
    Dataloader = Loaders()
    Dataloader.get_loaders(opt)
    training_data, validation_data, vist_train_data, vist_val_data = Dataloader.loader['train'], Dataloader.loader['val'], Dataloader.loader['vist_train'], Dataloader.loader['vist_val']
    #training_data, validation_data, vist_train_data, vist_val_data = Dataloader.loader['vist_train'], Dataloader.loader['vist_val'], Dataloader.loader['vist_train'], Dataloader.loader['vist_val']

    opt.src_vocab_size = len(Dataloader.frame_vocab)
    opt.tgt_vocab_size = len(Dataloader.story_vocab)

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    if opt.model:
        checkpoint = torch.load(opt.model)
        opt_ = checkpoint['settings']
        opt_.vist = True
        opt_.model=opt.model
        opt_.device=opt.device
        opt_.batch_size=opt.batch_size
        opt_.log=opt.log
        opt_.save_mode = opt.save_mode
        opt_.positional = opt.positional
        opt = opt_
        print(f"Load pretrain dic")

    device = torch.device(f'cuda:{opt.device}' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_encode_token_seq_len,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        positional = opt.positional).to(device)

    if opt.model:
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

    train(transformer, training_data, validation_data, vist_train_data, vist_val_data, optimizer, device ,opt, Dataloader)



if __name__ == '__main__':
    main()
