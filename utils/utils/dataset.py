import os
import logging
import random
import logging
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0
import glob
import json
import torch

def find_all_files(path, suffix=".jpg"):
    target_files = []
    for cur_dir, _, files in os.walk(path, followlinks=True):
        for f in files:
            if f.endswith(suffix):
                target_files.append(os.path.join(cur_dir, f))
    print_rank0(f'find {len(target_files)} files...')
    return target_files

class ItemDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
        super().__init__()
        self.data_dir = data_dirs
        self.data = self.load_data(data_dirs)
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
    
    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict
    
    def process_text(self, answer, prompt):
        return self.text_processor(answer, prompt)
    
    def load_data(self, data_dir):
        fname = glob.glob(os.path.join(data_dir, '*.json'))[0]
        with open(fname, "r") as f:
            all_files = json.load(f)['images']
        for sample in all_files:
            self.create_gt(sample)
        print_rank0(f"find {len(all_files)} samples in all...")
        return all_files

    def create_gt(self, data):
        def convert_bb(bbox):
            h, w = data['height'], data['width']
            temp = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
            temp = [int(t * 1000) for t in temp]
            return [temp[0], temp[1], temp[2] + temp[0], temp[3] + temp[1]]
        bboxes = []
        for bbox in data['bboxes']:
            bboxes.append(convert_bb(bbox['bbox']))
        result = []
        for rxn in data['reactions']:
            curr = {}
            for k, v in rxn.items():
                curr[k] = [[bboxes[i]] for i in v]
            result.append(curr)
        data['label'] = result
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # img
        try:
            img = Image.open(os.path.join(self.data_dir, data['file_name'])).convert('RGB')
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        img_dict = self.process_img(img)
        # text
        label = data['label']
        #random.shuffle(label)
        label = str(label)
        uni_key = label
        text_dict = self.process_text(label, 
            "Describe all reactions in the form [{'reactants': [[[x1,y1,x2,y2]], ... ], 'conditions': [[[x1,y1,x2,y2]], ... ]}, 'products': [[[x1,y1,x2,y2]], ... ]}, ... ]")
        if text_dict is None:
            print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
            return {}
        # other attr
        ret = {**img_dict, **text_dict, "question_id": uni_key}
        return ret