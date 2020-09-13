# vocabulary.py
import nltk
import pickle 
import os.path
from pycocotools.coco import COCO
from collections import Counter

class Vocabulary(object):
    def __init__(self, vocab_thres,
                 vocab_file = './data/vocab.pkl',
                 start_word = '<start>',
                 end_word = '<end>',
                 unk_word = '<unk>',
                 annotations_file = '../cocoapi/annotations/captions_train2017.json',
                 vocab_from_file = False):
        self.vocab_thres = vocab_thres
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()
    
    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""

        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as vf:
                vocab = pickle.load(vf)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('from pkl.file loaded!')
        
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as vf:
                pickle.dump(self, vf)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""

        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()
    
    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        coco = COCO(self.annotations_file)
        counter = Counter()
        idxs = coco.anns.keys()
        for id, idx in enumerate(idxs):
            caption = str(coco.anns[idx]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if id%100000 == 0 or id == (len(idxs) - 1):
                print(f'[{id}/{len(idxs)}] Tokenizing captions...')
        
        words = [word for word, cnt in counter.items() if cnt >= self.vocab_thres]

        for i, word in enumerate(words):
            self.add_word(word)
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)