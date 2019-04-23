# context_encoder.py

import torch, torch.nn as nn
from torch.autograd import Variable

class ContextRNN(nn.Module):
    def __init__(self, output_size, hidden_size, context_hidden_size):
        super(ContextRNN, self).__init__()
        self.hidden_size = hidden_size
        self.context_hidden_size = context_hidden_size

        self.gru = nn.GRU(hidden_size, context_hidden_size)

    def forward(self, input, hidden):
        output = input.view(1,1,-1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.context_hidden_size, device=device)