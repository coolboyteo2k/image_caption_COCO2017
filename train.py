from utils.read_json import process_data
from dataloader.data_loader import DataLoader
from model.model import EncoderCNN, DecoderRNN
from torchvision import transforms
from tqdm import tqdm
import skimage.io as io
import torch
import torch.utils.data as data
import torch.nn as nn
import nltk
import math
import os
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

#     captions_annFile_train = 'annotations/captions_train2017.json'
#     coco_caps = COCO(captions_annFile_train)
#     train_samples = process_data(captions_annFile_train)
#     assert len(os.listdir('./train2017')
#                ) == train_samples, 'Make sure to have full images in images/train2017'
# os.chdir('/content/drive/My Drive/opt/cocoapi/annotations')
nltk.download('punkt')
captions_annFile_train = 'annotations/captions_train2017.json'
coco_caps = COCO(captions_annFile_train)

# Number of images that have annos
train_samples = process_data(captions_annFile_train)

# -----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

BATCH_SIZE = 32
IMG_SIZE = 224

EMBED_SIZE = 300
HIDDEN_SIZE = 300

MIN_VOCAB_FREQS = 8
MAX_VOCAB_SIZE = 20000

data_transform = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    "val": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    "test": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
train_loader = DataLoader(transform=data_transform['train'], mode='train',
                          vocab_from_file=False, batch_size=BATCH_SIZE)

val_loader = DataLoader(transform=data_transform['val'], mode='val',
                        vocab_from_file=True, batch_size=BATCH_SIZE)


def visualize(data_loader):
    plt.figure(figsize=(15, 15))
    for i in range(2):
        plt.subplot(1, 2, i+1)
        rand = np.random.randint(train_samples)
        print(rand)
        item = data_loader.dataset[rand]

        img_show, caps = data_loader.dataset.imshow(item)
        plt.imshow(img_show)

        plt.title(caps)
        plt.axis('off')
    plt.show()

# visualize(train_loader)

# assert len(os.listdir('/train2017')
#            ) == train_samples, 'Make sure to have full images in images/train2017'


# prepare training
VOCAB_SIZE = len(train_loader.dataset.vocab)

encoder = EncoderCNN(EMBED_SIZE)  # .to(device)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE)  # .to(device)
print('Encoder:\n', encoder)
print('Decoder:\n', decoder)
all_params = list(encoder.linear.parameters()) + \
    list(encoder.bn1.parameters()) + list(decoder.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=all_params, lr=1e-05)
total_step = math.ceil(len(train_loader.dataset.cap_len) / BATCH_SIZE)

model_save_path = '/checkpoint'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path, exist_ok=True)


def train_predict(encoder, decoder):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        rand = np.random.randint(1000)
        batch = val_loader.dataset[rand]
        imgs = batch['image'].unsqueeze(0)  # .to(device)
        caps = batch['captions']

        img_show, _ = val_loader.dataset.imshow(batch)
        features = encoder(imgs).unsqueeze(1)
        output = decoder.test_sample(features)
        generated_cap = val_loader.dataset.display_generated_caption(
            output)

        plt.axis('off')
        plt.imshow(img_show)
        plt.show()
        print('Generated Caption: ', generated_cap)
        print('Ground Truth: ', caps)

    encoder.train()
    decoder.train()


num_epochs = 3


def train(encoder, decoder, optimizer, criterion, train_loader):
    running_loss = 0.0
    perplexity_thres = 7.5
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        for id in tqdm(range(0, total_step)):
            indices = train_loader.dataset.get_train_indices()
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            train_loader.batch_sampler.sampler = new_sampler
            batch_dict = next(iter(train_loader))

            encoder.zero_grad()
            decoder.zero_grad()

            imgs = batch_dict['image']  # .to(device)
            caps = batch_dict['captions_idx']

            # caps_target will not change when caps change - create a copy of caps
            caps_target = caps[:, :]  # .to(device)
            caps_train = caps  # .to(device)

            features = encoder(imgs)
            output = decoder(features, caps_train)

            loss = criterion(output.view(-1, VOCAB_SIZE),
                             caps_target.contiguous().view(-1))
            # perplexity is a metric for image caption
            perplexity = np.exp(loss.item())
            loss.backward()
            optimizer.step()

            if id % 300 == 0:
                print(
                    f'Epoch: [{epoch+1}]/[{num_epochs}] | Batch: [{id+1}]/[{total_step}]')
                print(
                    f'\tLoss: {loss.data: .4f} | Perplexity: {perplexity: .4f}')

                train_predict(encoder, decoder)

                if perplexity < perplexity_thres:
                    perplexity_thres = perplexity
                    torch.save(encoder.state_dict(
                    ), '/checkpoint/encoder_best.pt')
                    torch.save(decoder.state_dict(
                    ), '/checkpoint/decoder_best.pt')
                else:
                    torch.save(encoder.state_dict(
                    ), f'/checkpoint/encoder_resnet_{epoch+1}_{id+1}.pt')
                    torch.save(decoder.state_dict(
                    ), f'/checkpoint/decoder_resnet_{epoch+1}_{id+1}.pt')

                    print('Saved model!')

    return encoder, decoder


# Train
encoder, decoder = train(
    encoder, decoder, optimizer, criterion, train_loader)

# Load model
encoder.load_state_dict(torch.load(
    '/checkpoint/encoder_best.pt'))
decoder.load_state_dict(torch.load(
    '/checkpoint/decoder_best.pt'))

# Load test images + annos
test_loader = DataLoader(
    transform=data_transform['test'], mode='test', cocoapi='/content/drive/My Drive/opt')
data_iter = next(iter(test_loader))

# Prepare to plot test image


def get_sentences(original_img, all_preds):
    sentence = ' '
    plt.imshow(original_img.squeeze())

    return sentence.join([test_loader.dataset.vocab.idx2word[idx] for idx in all_preds[1:-1]])


def visualize_test_images(encoder, decoder):
    # encoder.to(device)
    # decoder.to(device)
    encoder.eval()
    decoder.eval()

    original_img, transform_img = next(data_iter)
    features = encoder(transform_img).unsqueeze(
        1)  # transform_img.to(device)
    final_output = decoder.test_sample(features, max_len=20)

    get_sentences(original_img, final_output)


visualize_test_images(encoder, decoder)
