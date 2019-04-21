# context_encoder.py

import torch, torch.nn as nn
from torch.autograd import Variable

class ContextRNN(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_size, n_layers=1):
        super(ContextRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        #self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #print input
        #output = self.embedding(input).view(1, 1, -1)
        output = input.view(1,1,-1)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)