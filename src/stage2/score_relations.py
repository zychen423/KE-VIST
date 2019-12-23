import pickle
import json
import torch
import torch.nn as nn
from model import RNNModel
from data_utils import FlattenDataset, lm_collate
from torch.utils.data import DataLoader
import math

BATCH_SIZE = 64

with open('/home/zychen/project/commen-sense-storytelling/relation_score/data/from_image2term_on_vist/storyid2images.json', 'r') as f:
    story_id2image_ids = json.load(f)
with open("/home/zychen/project/commen-sense-storytelling/relation_score/data/from_image2term_on_vist/img2terms.json") as f:
    img2terms = json.load(f)
with open("/home/zychen/project/commen-sense-storytelling/relation_score/data/from_image2term_on_vist/img_pair2path.vg.json") as f:
    img_pair2path = json.load(f)


def cudalize(x):
    return x.cuda()


def choose_path_policy(paths, ppls):
    print(len(possible_paths))
    print(len(ppls))
    return paths[ppls.index(min(ppls))]


def gen_candidate_paths(story_id, original_termsets):
    images = story_id2image_ids[story_id]
    candidate_paths = []
    for i in range(len(images)-1):
        first_img, second_img = images[i], images[i+1]
        possible_links = img_pair2path[f'{first_img}_{second_img}']
        for link in possible_links:
            new_terms = original_termsets[:]
            new_terms.insert(i, link)
            candidate_paths.append(new_terms)
    assert all([len(x) == 6 for x in candidate_paths])
    if len(candidate_paths) > 0:
        return candidate_paths
    else:
        return original_termsets


cross_entropy = nn.CrossEntropyLoss(reduction='none')


def loss_function(preds, labels, lens):
    new_preds, new_labels = [], []
    for pred, label, l in zip(preds, labels, lens):
        new_preds.append(pred[:l])
        new_labels.append(label[:l])
    preds = torch.cat(new_preds, dim=0)
    labels = torch.cat(new_labels, dim=0)
    losses = cross_entropy(preds, labels)
    start, end = 0, 0
    loss = []
    for l in lens:
        end = end + l
        loss.append(torch.min(losses[start:end]))
        start = start + l
    return loss


with open('/home/zychen/project/commen-sense-storytelling/relation_score/language_model_method/old_src/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
ntokens = tokenizer.vocab_size
model = RNNModel(ntokens, 100, 100, dropout=0.0,
                 pad_token=tokenizer.term2id['PAD'])
model = model.cuda()
model.load_state_dict(torch.load(
    '/home/zychen/project/commen-sense-storytelling/relation_score/language_model_method/old_src/model.pt'))
model = model.eval()
loss_fn = nn.CrossEntropyLoss(reduction='none')

original_termset2select_path = {}
used_cache = 0

result = {}
for count, (story_id, img_ids) in enumerate(story_id2image_ids.items()):
    if count % 100 == 1:
        with open('/home/zychen/project/commen-sense-storytelling/relation_score/language_model_method/old_src/vist_scored_terms_6_path.json', 'w') as f:
            print('saving until', count)
            json.dump(result, f, indent=4)
    temp_results = []
    original_termsets = [img2terms[img_id]
                         for img_id in img_ids]
    ppls = []

    if str(original_termsets) in original_termset2select_path:
        path = original_termset2select_path[str(original_termsets)]
        result[story_id] = path
        used_cache += 1
    else:
        possible_paths = gen_candidate_paths(story_id, original_termsets)
        dataset = FlattenDataset(possible_paths, tokenizer)
        dataloader = DataLoader(dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=0,
                                shuffle=False,
                                collate_fn=lambda x:
                                lm_collate(x, tokenizer.term2id['PAD']))
        for batch_num, batch in enumerate(dataloader):
            lm_input, lens, lm_output = batch
            predictions, _, lens = model(cudalize(lm_input), cudalize(lens))
            losses = loss_function(predictions, cudalize(lm_output), lens)
            ppls += [math.exp(loss.item()) for loss in losses]
            total_lens = len(dataloader)
            now_lens = min(batch_num*BATCH_SIZE, len(dataloader))
            print(
                f'\r{count}/{len(story_id2image_ids)} {now_lens}/{total_lens}, used_cache: {used_cache}', end='')

        path = choose_path_policy(possible_paths, ppls)
        original_termset2select_path[str(original_termsets)] = path
        result[story_id] = path
    print()
with open('/home/zychen/project/commen-sense-storytelling/relation_score/language_model_method/old_src/vist_scored_terms_6_path.json', 'w') as f:
    json.dump(result, f, indent=4)
