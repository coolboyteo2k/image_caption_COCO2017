import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn1 = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn1(features)

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device=None, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = device
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)) \
            if self.device is not None else (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                             torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, features, captions):
        captions = captions[:, :-1]
        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        embeds = self.word_embeddings(captions)
        inputs = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.linear(lstm_out)

        return outputs

    def test_sample(self, inputs, max_len=20):
        final_output = []
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)

        while len(final_output) < max_len:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_idx = torch.max(outputs, dim=1)
            final_output.append(max_idx.cpu().numpy()[0].item())

            inputs = self.word_embeddings(max_idx)
            inputs = inputs.unsqueeze(1)

        return final_output
