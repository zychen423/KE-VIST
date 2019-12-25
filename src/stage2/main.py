import json
import pickle
import math
import torch
import torch.nn as nn
from data_utils import Tokenizer, FlattenDataset, split_dataset, lm_collate
from torch.utils.data import DataLoader
from model import RNNModel


NUM_WORKERS = 0


def cudalize(x):
    return x.cuda()


ROC_terms_path = './data/ROC_replace_coref_gender_mapped_frame_noun.json'
termset_seqs = []
with open(ROC_terms_path, 'r') as f:
    objs = json.load(f)
    for obj in objs:
        termset_seq = obj['gender_coref_mapped_seq']
        termset_seqs.append(termset_seq)
VIST_terms_path = './data/VIST_coref_nos_mapped_frame_noun_train.json'
with open(VIST_terms_path, 'r') as f:
    termsets = json.load(f)
    termset_seq = []
    for i, termset in enumerate([x['coref_mapped_seq'] for x in termsets]):
        termset_seq.append(termset)
        if i % 5 == 4:
            termset_seqs.append(termset_seq)
            termset_seq = []

tokenizer = Tokenizer(termset_seqs)
with open('./tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
ntokens = tokenizer.vocab_size
print('vocab_size', ntokens)
dataset = FlattenDataset(termset_seqs, tokenizer)
train_dataset, valid_dataset, test_dataset = split_dataset(dataset)
print(
    f'train:{len(train_dataset)} valid:{len(valid_dataset)} \
        test:{len(test_dataset)}'
)
train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          num_workers=NUM_WORKERS,
                          shuffle=True,
                          collate_fn=lambda x:
                          lm_collate(x, tokenizer.term2id['PAD']))
valid_loader = DataLoader(valid_dataset,
                          batch_size=64,
                          num_workers=NUM_WORKERS,
                          collate_fn=lambda x:
                          lm_collate(x, tokenizer.term2id['PAD']))
test_loader = DataLoader(test_dataset,
                         batch_size=64,
                         num_workers=NUM_WORKERS,
                         collate_fn=lambda x:
                         lm_collate(x, tokenizer.term2id['PAD']))

model = RNNModel(ntokens, 100, 100, dropout=0.0,
                 pad_token=tokenizer.term2id['PAD'])
model = cudalize(model)
loss_fn = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0)


cross_entropy = nn.CrossEntropyLoss()


def loss_function(preds, labels, lens):
    # TODO: delete padding
    new_preds, new_labels = [], []
    for pred, label, l in zip(preds, labels, lens):
        new_preds.append(pred[:l])
        new_labels.append(label[:l])
    preds = torch.cat(new_preds, dim=0)
    labels = torch.cat(new_labels, dim=0)
    return cross_entropy(preds, labels)


earlystop_count, min_loss = 0, 100
for epoch in range(0, 10000):
    total_loss = 0
    model = model.train()
    for batch_num, batch in enumerate(train_loader):
        # input: BOS, term0,
        # target: term0, term1
        lm_input, lens, lm_output = batch
        predictions, _, lens = model(cudalize(lm_input), cudalize(lens))
        loss = loss_function(predictions, cudalize(lm_output), lens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / (batch_num + 1)
        ppl = math.exp(avg_loss)
        print(
            f'\rLM Training: epoch:{epoch} train batch:{batch_num} '
            + f'loss:{avg_loss} ppl:{ppl}',
            end='')
    print()
    valid_loss = 0
    model = model.eval()
    for batch_num, batch in enumerate(valid_loader):
        lm_input, lens, lm_output = batch
        predictions, _, lens = model(cudalize(lm_input), cudalize(lens))
        loss = loss_function(predictions, cudalize(lm_output), lens)
        valid_loss += loss.item()
        valid_avg_loss = valid_loss / (batch_num + 1)
        valid_ppl = math.exp(valid_avg_loss)
        print(f'\rvalid batch:{batch_num} loss: {valid_avg_loss:.4f} '
              + f'ppl:{valid_ppl:.4f}', end='')
    print()
    if valid_loss / (batch_num + 1) < min_loss:
        min_loss = valid_loss / (batch_num + 1)
        earlystop_count = 0
        torch.save(model.state_dict(), './model.pt')
        print('saved model')
    else:
        earlystop_count += 1
        if earlystop_count > 20:
            print('earlystop')
            break
    print()
test_loss = 0
for batch_num, batch in enumerate(test_loader):
    lm_input, lens, lm_output = batch
    predictions, _, lens = model(cudalize(lm_input), cudalize(lens))
    loss = loss_function(predictions, cudalize(lm_output), lens)
    test_loss += loss.item()
    test_avg_loss = test_loss / (batch_num + 1)
    test_ppl = math.exp(test_avg_loss)
    print(f'\rtest batch:{batch_num} loss: {test_avg_loss:.4f} '
          + f'ppl:{test_ppl:.4f}', end='')
