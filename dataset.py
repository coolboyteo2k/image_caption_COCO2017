import nltk
import os
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm.notebook import tqdm
import random
import matplotlib.pyplot as plt
import json
import pickle
import os.path

from vocab import Vocabulary


class CoCoDataset(data.Dataset):
    def __init__(self, vocab_file, vocab_thres, start_word,
                 end_word, unk_word, annotations, vocab_from_file, img_folder, batch_size, mode, transform=None):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.img_folder = img_folder

        self.vocab = Vocabulary(vocab_thres, vocab_file, start_word,
                                end_word, unk_word, annotations, vocab_from_file)

        if self.mode == 'train' or self.mode == 'val':
            self.coco = COCO(annotations)
            self.idxs = list(self.coco.anns.keys())
            vocab_tokens = [nltk.tokenize.word_tokenize(str(
                self.coco.anns[self.idxs[id]]['caption']).lower()) for id in tqdm(np.arange(len(self.idxs)))]
            self.cap_len = [len(token) for token in vocab_tokens]

        else:
            test_load = json.load(open(annotations).read())
            self.paths = [path['file_name'] for path in paths['images']]

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val':
            anno_id = self.idxs[idx]
            caption = self.coco.anns[anno_id]['caption']
            img_id = self.coco.anns[anno_id]['image_id']
            # 'file_name': '000000380706.jpg'
            path = self.coco.loadImgs(img_id)[0]['file_name']

            image = Image.open(os.path.join(
                self.img_folder, path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

            # Convert caption to tensor of word idxs
            tokens = nltk.tokenize.word_tokenize(
                str(caption).lower())  # list of 1 caption-token or more

            tokens_idx = []
            tokens_idx.append(self.vocab(self.vocab.start_word))
            tokens_idx.extend([self.vocab(token) for token in tokens])
            tokens_idx.append(self.vocab(self.vocab.end_word))

            tokens_len = torch.Tensor([len(tokens_idx)])
            tokens_idx = torch.LongTensor(tokens_idx)

            img_dict = {}
            img_dict['image'] = image
            img_dict['captions'] = ' '.join(tokens)
            img_dict['captions_idx'] = tokens_idx
            img_dict['cap_len'] = tokens_len
            return img_dict

        else:
            path = self.paths[idx]
            PIL_img = Image.open(os.path.join(
                self.img_folder, path)).convert('RGB')
            orig_img = np.array(PIL_img)
            image = self.transform(PIL_img)
            return orig_img, image

    def display_generated_caption(self, output):
        caption = []
        for i in output:
            caption.append(self.vocab.itos(i))
            # if i == self.vocab(self.vocab.end_word):
            #     break

        caption = ' '.join(caption[1:-1])
        caption = caption.capitalize()
        return caption

    def imshow(self, img_dict):
        img = img_dict['image']
        caps = img_dict['captions']
        idx = img_dict['captions_idx']
        length = img_dict['cap_len']
        if self.mode == 'val':
            img = img.squeeze()
        np_img = np.array(img).transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_img = np_img*std + mean
        img_show = np.clip(np_img, 0, 1)

        return img_show, caps

    def get_train_indices(self):
        sel_length = np.random.choice(self.cap_len)
        all_indices = np.where(
            [self.cap_len[i] == sel_length for i in np.arange(len(self.cap_len))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train' or self.mode == 'val':
            return len(self.idxs)
        else:
            return len(self.paths)
