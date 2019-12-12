from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse
import json

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--output_dir', default='vist_data_fix_rotation/vist', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'classes', 'attributes', 'attr_gt_thresh']

data_path = './data/genome/1600-400-20'

# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())
print (os.getcwd())
# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())

def get_class_attr(cls, attr, attr_gt_thresh):
    object_list = []
    for i in range(len(cls)):
        c = classes[cls[i]+1]
        if attr_gt_thresh[i]==True:
            c = attributes[attr[i]+1] + " " + c
        object_list.append(c)
    return object_list

infiles = ['./vist_with_classes_attr.csv']

os.makedirs(args.output_dir+'_att')
os.makedirs(args.output_dir+'_fc')
os.makedirs(args.output_dir+'_box')

img2obj = {}
count = 0
check_list = set(os.listdir('./vist_data_fix_rotation/VG_box/'))
print('check list processed')
for infile in infiles:
    print('Reading ' + infile)
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = str(item['image_id'].split('.')[0])
            cls = np.frombuffer(base64.b64decode(item['classes']), dtype=np.int64)
            attr = np.frombuffer(base64.b64decode(item['attributes']), dtype=np.int64)
            attr_gt_thresh = np.frombuffer(base64.b64decode(item['attr_gt_thresh']), dtype=np.bool)
            if item['image_id']+'.npy' not in check_list:
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.b64decode(item[field]),
                            dtype=np.float32).reshape((item['num_boxes'],-1))
                np.savez_compressed(os.path.join(args.output_dir+'_att', str(item['image_id'])), feat=item['features'])
                np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
                np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])
            img2obj[str(item['image_id'])] = get_class_attr(cls, attr, attr_gt_thresh)
            count+=1
            if count%1000 == 0:
                print(count, 'files')
json.dump(img2obj, open(args.output_dir+'_img2obj_dic.json', 'w'), indent=4)
