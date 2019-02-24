# decoder.py

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

######################################################################
# Attention Decoder
# -----------------
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.

class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_size, max_length, dropout_p=0.1):
        print('decoder, max_len={}'.format(max_length))
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.attn = nn.Linear(self.hidden_size + self.embedding_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.embedding_size, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.context_transform = nn.Linear(self.hidden_size + self.embedding_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, context):
        print(f'input size:            {input.size()}')
        print(f'hidden size:           {hidden.size()}')
        for i, encoder_output in enumerate(encoder_outputs):    
            print(f'encoder output[{i}]:   {encoder_output.size()}')
        print(f'context size:          {context.size()}')

        #print(f'input: {input}')
        #print(f'self.embedding: {self.embedding}')
        #print(f'self.embedding(input).view(1, 1, -1): {self.embedding(input).view(1, 1, -1)}')
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        print(f'embedded[0]: {embedded[0].size()}')
        print(f'hidden[0]:   {hidden[0].size()}')
        print(f'torch.cat size: {torch.cat((embedded[0], hidden[0]), 1).size()}')
        #print(f'torch.cat:      {torch.cat((embedded[0], hidden[0]), 1)}')
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        # NEW: output for context related additions
        # inputs are concatenation of previous output and context
        #for i in range(self.n_layers): #NOTE: more than one layer is not implemented yet?
        output = F.relu(output)
        print(f'Output[0] size:   {output.size()}')
        print(f'Context[0] size:  {context.size()}')
        print(f'linear transform: {self.context_transform(torch.cat((output[0], context[0]), 1)).size()}')
        #output = torch.cat((output, context), 0)
        # Added: transforming a 256+300-d tensor into 300-d tensor for the GRU
        output = self.context_transform(torch.cat((output[0], context[0]), 1)) # NOTE: not sure if this is correct
        output, hidden = self.gru(output, hidden)

        """ previous output for just decoer
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        """

        # log softmax is done for NLL Criterion. We could use CrossEntropyLoss to avoid calculating this
        # log_softmax uses log base e
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result