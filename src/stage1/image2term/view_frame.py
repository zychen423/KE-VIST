# coding: utf-8
import ROC_loader_manager
from build_roc_frame_vocab import *

def takeSecond(elem):
    return elem[1]

loader = ROC_loader_manager.Loaders()
verb_frames=[]
for i in range(8,len(loader.frame_vocab)):
    word = loader.frame_vocab.idx2word[i]
    if word.split('_')[1]=="NOUN":
        verb_frames.append((word, loader.frame_vocab.word_count[i]))

verb_frames = sorted(verb_frames, key=takeSecond, reverse=True)
print(verb_frames[:20])

