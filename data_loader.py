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
from dataset import CoCoDataset


def DataLoader(vocab_file='cocoapi/data/vocab.pkl', vocab_thres=8, start_word='<start>',
               end_word='<end>', unk_word='<unk>', vocab_from_file=True, cocoapi='/opt',
               num_workers=0, pin_memory=False, batch_size=32, mode='train', transform=None):

    assert mode in ['train', 'test', 'val']
    if vocab_from_file == False:
        assert mode == 'train', 'If False, create vocab from scratch & override any existing vocab_file'

    if mode == 'train':
        if vocab_from_file == True:
            assert os.path.exists(
                vocab_file),  'If True, load vocab from from existing vocab_file, if it exists'
        img_folder = os.path.join(cocoapi, 'cocoapi/images/train2017')
        annotations = os.path.join(
            cocoapi, 'cocoapi/annotations/captions_train2017.json')

    elif mode == 'test':
        assert os.path.exists(vocab_file), 'vocab.pkl must exist'
        assert vocab_from_file == True, 'change to True'
        img_folder = os.path.join(cocoapi, 'cocoapi/images/test2017')
        annotations = os.path.join(
            cocoapi, 'cocoapi/annotations/image_info_test2017.json')

    elif mode == 'val':
        img_folder = os.path.join(cocoapi, 'cocoapi/images/val2017')
        annotations = os.path.join(
            cocoapi, 'cocoapi/annotations/captions_val2017.json')

    vocab_file = os.path.join(cocoapi, vocab_file)
    dataset = CoCoDataset(vocab_file, vocab_thres, start_word, end_word,
                          unk_word, annotations, vocab_from_file, img_folder,
                          batch_size, mode, transform)
    print(f'Number of unique words in the dataset: {len(dataset.vocab)}')
    if mode == 'train':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_loader = data.DataLoader(dataset=dataset, num_workers=num_workers,
                                       pin_memory=pin_memory,
                                       batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                               batch_size=dataset.batch_size,
                                                                               drop_last=False))
        return train_loader

    elif mode == 'test':
        test_loader = data.DataLoader(dataset=dataset, batch_size=dataset.batch_size,
                                      shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        return test_loader

    elif mode == 'val':
        val_loader = data.DataLoader(dataset=dataset, batch_size=1,
                                     num_workers=num_workers, pin_memory=pin_memory)
        return val_loader
