# hred
import copy, math, operator, os, random, re, time
from pathlib import Path

import pymeteor.pymeteor as pymeteor

import torch, torch.nn as nn
from torch import optim
from torch.autograd import Variable

from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from storygen import book, encoder, decoder, glove, log
from storygen.book import START_ID, STOP_ID
from storygen.encoder import EncoderRNN
from storygen.decoder import DecoderRNN
from storygen.context_encoder import ContextRNN

CHECKPOINT_FILE_FORMAT = '{}/model_{}_{}_{}_{}_{}.torch'

CHECKPOINT_FORMAT = 'model_(\d+)_{}_{}_{}_{}.torch'
CHECKPOINT_DIR = 'obj'

LOSS_FILE_FORMAT = '{}loss.dat'
VALIDATION_LOSS_FILE_FORMAT = '{}validation.dat'

# State dicts fields
ENCODER_STATE_DICT = 'encoder_state_dict'
DECODER_STATE_DICT = 'decoder_state_dict'
CONTEXT_STATE_DICT = 'context_state_dict'
ENCODER_OPTIMIZER  = 'encoder_optimizer'
DECODER_OPTIMIZER  = 'decoder_optimizer'
CONTEXT_OPTIMIZER  = 'context_optimizer'
OPTIMIZER_TYPE     = 'optimizer_type'
CONTEXT_HIDDEN     = 'context_hidden'

OPTIMIZER_TYPES = ['adam', 'sgd']

## HELPER FUNCTIONS ##
# Converts a sentence into a list of indexes
def indexesFromSentence(book, sentence):
    return [book.word2index[word] for word in sentence]

# Converts an index (integer) to a pytorch tensor
def tensorFromIndex(index, device):
    return torch.tensor(index, dtype=torch.long, device=device).view(-1, 1)

# Converts a sentence to a pytorch tensor
def tensorFromSentence(book, sentence, device):
    indexes = indexesFromSentence(book, sentence)
    indexes.append(STOP_ID)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromParagraph(book, paragraph, device):
    return [tensorFromSentence(book, sentence, device) for sentence in paragraph]


# Calculates the BLEU score via NLTK
def calculateBleu(candidate, reference, n_gram=2):
    # looks at ration of n-grams between 2 texts
    # Break candidate/reference into the format below
    candidate = candidate.split()
    reference = reference.split()
    return sentence_bleu([reference], candidate)#, weights=(1,0,0,0))
    
# Helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '{}m {:.2f}s'.format(m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '{} (- {})'.format(asMinutes(s), asMinutes(rs))

# Object used in Beam Search to keep track of the results at each depth of the search tree
class BeamSearchResult:
        # score:   float of the sentence's current score
        # item:    the item (or list of items) to create the sentence with
        # hidden:  the current decoder hidden layer
        # stopped: has the sentence reached EOL or not?
        # words:   the items decoded into their corresponding strings
        def __init__(self, score, item, hidden):
                if isinstance(item, list):
                        self.items = item
                else:
                        self.items = [item]
                self.score = score
                self.hidden = hidden
                self.stopped = False
                self.words = []
        # Create a new BeamSearchResult with the values of 'self' and 'result'
        def add_result(self, result):
                new_result = BeamSearchResult(self.score + result.score, self.items + result.items, result.hidden)
                new_result.words += self.words
                return new_result
        # Returns the last element in the items list
        def get_latest_item(self):
                return self.items[-1]
        # Performs the perplexity calculation where summation = score and N = length of items
        def calculate_perplexity(self):
                return pow(math.e, -self.score / len(self.items))
        def __repr__(self):
                return 'BeamSearchResult: score={:.4f}, stopped={}, words="{}"'.format(self.score, str(self.stopped), ' '.join(self.words))

######################################################################
# The HRED Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.

# Represents a Hierarchical Recurrent Encoder-Decoder that's made up of
# a sentence-encoder RNN, context-encoder RNN, and a decoder RNN with 
# attention weights
class Hred(object):
    def __init__(self, device, book, max_length, max_context, hidden_size, context_hidden_size, embedding_size, 
            optimizer_type='adam',
            use_context_attention=False,
            teacher_forcing_ratio=0.5,
            beam=5,
            context_layers = 1,
            attention_layers = 1,
            decoder_layers = 1,
            learning_rate = 0.0001,
            ):
        # Original parameters
        self.book = book
        self.hidden_size = hidden_size
        self.context_hidden_size = context_hidden_size
        self.max_length = max_length
        self.max_context = max_context
        self.embedding_size = embedding_size

        self.encoder = None
        self.decoder = None
        self.context = None

        self.context_hidden = None
        self.use_context_attention = use_context_attention

        self.device = device

        # New parameters
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam = beam
        self.context_layers = context_layers
        self.attention_layers = attention_layers
        self.decoder_layers = decoder_layers
        self.learning_rate = learning_rate
        
        self.optimizer_type = optimizer_type
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.context_optimizer = None

        self.log = log.Log()

    def loadFromFiles(self, checkpoint_filename):
        # Check that the path for the file exists
        os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)
        
        checkpoint_file = Path(checkpoint_filename)

        if checkpoint_file.is_file():
            print("Loading model and context from files...")
            checkpoint = torch.load(checkpoint_filename)

            # Load optimizer type
            self.optimizer_type = checkpoint[OPTIMIZER_TYPE]

            # Load encoder
            self.encoder = EncoderRNN(self.book.n_words, self.hidden_size, self.embedding_size).to(self.device)
            self.encoder.load_state_dict(checkpoint[ENCODER_STATE_DICT])
            if self.optimizer_type == 'adam':
                self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == 'sgd':
                self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
            self.encoder_optimizer.load_state_dict(checkpoint[ENCODER_OPTIMIZER])
            
            # Load decoder
            self.decoder = DecoderRNN(self.book.n_words, self.hidden_size, self.context_hidden_size, 
                                        self.embedding_size, self.max_length, self.max_context,
                                        use_context_attention=self.use_context_attention).to(self.device)
            self.decoder.load_state_dict(checkpoint[DECODER_STATE_DICT])
            if self.optimizer_type == 'adam':
                self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == 'sgd':
                self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.learning_rate)
            self.decoder_optimizer.load_state_dict(checkpoint[DECODER_OPTIMIZER])

            # Load context encoder
            self.context = ContextRNN(self.book.n_words, self.hidden_size, self.context_hidden_size).to(self.device)
            self.context.load_state_dict(checkpoint[CONTEXT_STATE_DICT])
            if self.optimizer_type == 'adam':
                self.context_optimizer = optim.Adam(self.context.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == 'sgd':
                self.context_optimizer = optim.SGD(self.context.parameters(), lr=self.learning_rate)
            self.context_optimizer.load_state_dict(checkpoint[CONTEXT_OPTIMIZER])

            self.context_hidden = checkpoint[CONTEXT_HIDDEN].to(self.device)

            return True
        return False

    def saveToFiles(self, checkpoint_filename):
        # Check that the path for both files exists
        os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)
        
        checkpoint_file = Path(checkpoint_filename)

        checkpoint_state = {
            OPTIMIZER_TYPE: self.optimizer_type,
            ENCODER_STATE_DICT: self.encoder.state_dict(),
            ENCODER_OPTIMIZER: self.encoder_optimizer.state_dict(),
            DECODER_STATE_DICT: self.decoder.state_dict(),
            DECODER_OPTIMIZER: self.decoder_optimizer.state_dict(),
            CONTEXT_STATE_DICT: self.context.state_dict(),
            CONTEXT_OPTIMIZER: self.context_optimizer.state_dict(),
            CONTEXT_HIDDEN: self.context_hidden
        }

        torch.save(checkpoint_state, checkpoint_file)
        
    def _train(self, input_variable, target_variable,
            encoder_model, decoder_model, context_model,
            context_hidden, context_outputs,
            encoder_optimizer, decoder_optimizer, 
            criterion, last):
        
        encoder_hidden = encoder_model.initHidden(self.device)

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = torch.zeros(self.max_length, encoder_model.hidden_size, device=self.device)

        loss = 0

        # Encode the input sentence
        for ei in range(input_length):
            if ei < self.max_length:
                encoder_output, encoder_hidden = encoder_model(input_variable[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0][0]
            else:
                print(f'Somehow we got ei={ei} for range({input_length}) where max_length={self.max_length}')

        # NOTE: my method ()
        
        # The "sentence vector" is the hidden state obtained after the last token of the sentence has been processed
        # encoder_hidden = sentence_vector
        # The Context RNN keeps track of past sentences by processing iteratively each sentence vector
        context_output, context_hidden = context_model(encoder_hidden, context_hidden)
        # After processing sentence S_n, the hidden state of the context RNN represents a summary of the sentences up to and
            # including sentence n, which is used to predict the next sentence S_n+1
        # This hidden state can be interpreted as the continuous-valued state of the dialogue system
        # See context_hidden, context_output will be given as input to next sentence

        # The next sentence prediction is performed by means of a decoder RNN
        # Takes the hidden state of the context RNN and produces a probability distribution over the tokens in the next sentence
        # Its prediction is conditioned on the hidden state of the context RNN
        decoder_input = torch.tensor([[START_ID]], device=self.device)

        decoder_hidden = None 
        
        ## End my method

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder_model(
                        decoder_input, decoder_hidden, encoder_outputs, context_hidden, context_outputs)
                
                if last:
                    #loss += criterion(decoder_output[0], target_variable[di]) #NOTE: this is how it is in other project?
                    loss += criterion(decoder_output, target_variable[di]) # previous project
                    #print(f'decoder_output: {decoder_output}\ntarget_variable[{di}]: {target_variable[di]}')
                decoder_input = target_variable[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder_model(
                    decoder_input, decoder_hidden, encoder_outputs, context_hidden, context_outputs)
                topv, topi = decoder_output.topk(1)

                # from seq2seq project, .detach() is NECESSARY for loss.backward() (?)
                decoder_input = topi.squeeze().detach() # detach from history as input

                if last:
                    loss += criterion(decoder_output, target_variable[di]) # previous project
                if decoder_input.item() == STOP_ID:
                    break

        if last:
            # NOTE: moved outside of loop
            loss.backward()
            # NOTE: We need to detach the "hidden state" between "batches"?
            # Trying:
            context_hidden = context_hidden.detach()
            
            return loss.data.item() / target_length, context_hidden
        else:
            return context_hidden

    def _train_paragraph(self, paragraph,
            encoder_model, decoder_model, context_model,
            context_hidden, 
            encoder_optimizer, decoder_optimizer, context_optimizer,
            criterion):
        
        tensors = tensorsFromParagraph(self.book, paragraph, self.device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        context_optimizer.zero_grad()

        context_outputs = torch.zeros(self.max_context, self.context_hidden_size, device=self.device)

        for i in range(len(tensors)-1):
            input_variable = tensors[i]
            target_variable = tensors[i+1]
            last = False
            if i+1 == len(tensors)-1:
                last = True
            if last:
                loss, context_hidden = self._train(input_variable, target_variable, encoder_model, decoder_model, context_model,
                        context_hidden, context_outputs, encoder_optimizer, decoder_optimizer, criterion, last)

                encoder_optimizer.step()
                decoder_optimizer.step()
                context_optimizer.step()

                return loss, context_hidden
            else:
                context_hidden = self._train(input_variable, target_variable, encoder_model, decoder_model, context_model,
                        context_hidden, context_outputs, encoder_optimizer, decoder_optimizer, criterion, last)
            context_outputs[i] = context_hidden
        print('We reached a part of training that should be unreachable! Empty paragraph perhaps?')
        print(f'Paragraph: {paragraph}')
        exit(1)

    ######################################################################
    # The whole training process looks like this:
    #
    # -  Start a timer
    # -  Initialize optimizers and criterion
    # -  Create set of training pairs
    # -  Start empty losses array for plotting
    #
    # Then we call "train" many times and occasionally print the progress (%
    # of examples, time so far, estimated time) and average loss.
    def train_model(self, epochs, train_paragraphs, validation_paragraphs, validate_every=3, # validation_size=0.1,
            embedding_type=None, save_temp_models=False, checkpoint_every=25, loss_dir=None,
            print_every=10, plot_every=100, evaluate_every=500):
        global CHECKPOINT_DIR
        logfile = self.log.create('seq2seq-train-model')
        # TODO: re-add checkpoints and embeddings
        
        # Set folder for checkpoints
        if embedding_type is not None:
            if embedding_type == 'glove':
                CHECKPOINT_DIR = 'obj_glove'
            elif embedding_type == 'sg':
                CHECKPOINT_DIR = 'obj_sg'
            elif embedding_type == 'cbow':
                CHECKPOINT_DIR = 'obj_cbow'
            else:
                print('Incorrect embedding type given! Please choose one of ["glove", "sg", "cbow"]')
                exit()

        self.log.info(logfile, 'Training for {} epochs'.format(epochs))
        self.log.info(logfile, 'Embedding type: {}'.format(embedding_type))

        # Check if any checkpoints for this model exist:
        checkpoints = set()

        for filename in os.listdir('{}/'.format(CHECKPOINT_DIR)):
            checkpoint = re.search(CHECKPOINT_FORMAT.format(
                self.embedding_size, self.hidden_size, self.max_length, self.optimizer_type), filename)
            if checkpoint:
                checkpoints.add(int(checkpoint.group(1)))
        
        print('Checkpoints found at: {}'.format(checkpoints))
        self.log.debug(logfile, 'Checkpoints found at: {}'.format(checkpoints))
        start_epoch = 0

        found_max_checkpoint = False
        while not found_max_checkpoint:
            if len(checkpoints) > 0:
                max_val = max(checkpoints)
                if max_val < epochs:
                    start_epoch = max_val
                    print('Found checkpoint at epoch={}'.format(start_epoch))
                    self.log.debug(logfile, 'Found checkpoint at epoch={}'.format(start_epoch))
                    found_max_checkpoint = True
                else:
                    checkpoints.remove(max_val)
            else:
                found_max_checkpoint = True # the max is 0 (none exists)
                print('No checkpoint found')
                self.log.debug(logfile, 'No checkpoint found')

        loss_avgs = []
        validation_loss_avgs = []
        
        # If we didn't load the encoder/decoder from files: create new ones or load checkpoint to train
        if self.encoder is None or self.decoder is None or self.context is None:
            
            if start_epoch > 0:
                # Load the encoders/decoder for the starting epoch checkpoint
                checkpoint_filename = CHECKPOINT_FILE_FORMAT.format(
                    CHECKPOINT_DIR, start_epoch, self.embedding_size, self.hidden_size, self.max_length, self.optimizer_type)
                if self.loadFromFiles(checkpoint_filename):#encoder_filename, decoder_filename, context_filename):
                    self.log.info(logfile, 'Loaded model from file at checkpoint {}'.format(start_epoch))
                else:
                    self.log.error(logfile, 'Tried to load checkpoint encoder/decoder at epoch={}, but it failed!'.format(start_epoch))
                    print('Checkpoint loading error!')
                    exit(1)
                # Load the loss values from files, if given
                if loss_dir is not None:
                    # Add a forward slash to end of directory path
                    if not loss_dir[-1] == '/':
                        loss_dir += '/'
                    self.log.info(logfile, 'Attempting to load loss files from {}'.format(loss_dir))
                    loss_filename = LOSS_FILE_FORMAT.format(loss_dir)
                    validation_loss_filename = VALIDATION_LOSS_FILE_FORMAT.format(loss_dir)
                    # Add (epoch, loss value) pairs to the loss lists
                    if os.path.isfile(loss_filename):
                        self.log.debug(logfile, 'Loading loss file: {}'.format(loss_filename))
                        print('Loading loss file: {}'.format(loss_filename))
                        with open(loss_filename, 'r') as f:
                            for line in f.readlines(): # should just have one line
                                for pair in line.split('\t'):
                                    if len(pair) > 0:
                                        epoch, value = pair.strip().split(',', 1)
                                        loss_avgs.append((int(epoch), float(value)))
                        # Save the values in the new log directory
                        with open(LOSS_FILE_FORMAT.format(self.log.dir), 'w+') as f:
                            for item in loss_avgs:
                                f.write('{},{}\t'.format(item[0], item[1]))
                    if os.path.isfile(validation_loss_filename):
                        self.log.debug(logfile, 'Loading validation loss file: {}'.format(validation_loss_filename))
                        print('Loading validation loss file: {}'.format(validation_loss_filename))
                        with open(validation_loss_filename, 'r') as f:
                            for line in f.readlines(): # should just have one line
                                for pair in line.split('\t'):
                                    if len(pair) > 0:
                                        epoch, value = pair.strip().split(',', 1)
                                        validation_loss_avgs.append((int(epoch), float(value)))
                        # Save the values in the new log directory
                        with open(VALIDATION_LOSS_FILE_FORMAT.format(self.log.dir), 'w+') as f:
                            for item in validation_loss_avgs:
                                f.write('{},{}\t'.format(item[0], item[1]))
            else:
                self.encoder = EncoderRNN(self.book.n_words, self.hidden_size, self.embedding_size).to(self.device)
                self.decoder = DecoderRNN(self.book.n_words, self.hidden_size, self.context_hidden_size, 
                                            self.embedding_size, self.max_length, self.max_context,
                                            use_context_attention=self.use_context_attention).to(self.device)
                self.context = ContextRNN(self.book.n_words, self.hidden_size, self.context_hidden_size).to(self.device)

                if self.optimizer_type == 'adam':
                    self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
                    self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
                    self.context_optimizer = optim.Adam(self.context.parameters(), lr=self.learning_rate)
                elif self.optimizer_type == 'sgd':
                    self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
                    self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.learning_rate)
                    self.context_optimizer = optim.SGD(self.context.parameters(), lr=self.learning_rate)
                
                self.context_hidden = self.context.initHidden(self.device)
                print(f'opt type: {self.optimizer_type}')
                self.context_optimizer.zero_grad()
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
        
        # Create the GloVe embedding's weight matrix:
        if embedding_type is not None:
            # Generates a dict of a word to its GloVe vector
            words2vec = glove.generate_glove(dim_size=self.embedding_size, embedding_type=embedding_type)
            # Create weight matrix:
            weights_matrix = np.zeros((self.book.n_words, self.embedding_size))
            words_found = 0
            for word in self.book.word2index:
                idx = self.book.word2index[word]
                try:
                    weights_matrix[idx] = words2vec[word]
                    words_found += 1
                except KeyError:
                    # Create random vector of dimension 'embedding_size', scale=0.6 taken from tutorial
                    weights_matrix[idx] = np.random.normal(scale=0.6, size=(self.embedding_size, ))
            # Convert weights_matrix to a Tensor
            weights_matrix = torch.tensor(weights_matrix, device=self.device)

            print('We found {}/{} words in our GloVe words2vec dict!'.format(words_found, self.book.n_words))
            self.log.info(logfile, 'Found {}/{} words in the GloVe dict.'.format(words_found, self.book.n_words))
            # Set the embedding layer's state_dict for encoder and decoder
            self.encoder.embedding.load_state_dict({'weight': weights_matrix})
            self.decoder.embedding.load_state_dict({'weight': weights_matrix})
            self.log.info(logfile, 'Created encoder and decoder embeddings')
        
        start = time.time()
        print_loss_total = 0  # Reset every print_every

        # TODO: rearrange above loading/checkpoint code
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        criterion = nn.NLLLoss()

        iter = 0
        # Iterate through the training set over a set amount of epochs
        # Output the progress and current loss value
        print(f'Training size: {len(train_paragraphs)}')
        for i in range(start_epoch, epochs+1):
            print(f'Processing epoch {i}...')
            self.log.debug(logfile, 'Processing epoch {}'.format(i))

            loss_avg = 0
            for j, train_paragraph in enumerate(train_paragraphs):
                iter += 1

                loss, self.context_hidden = self._train_paragraph(train_paragraph,
                    self.encoder, self.decoder, self.context,
                    self.context_hidden, self.encoder_optimizer, self.decoder_optimizer, self.context_optimizer, criterion)

                loss_avg += loss
                print_loss_total += loss
                if j % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0

                    epochs_processed = (i-start_epoch)*len(train_paragraphs)+j

                    progress_percent = epochs_processed / ((epochs-start_epoch)*len(train_paragraphs))
                    t = -1.0
                    if progress_percent > 0:
                        t = timeSince(start, progress_percent)
                    print('{} ({} {:.2f}%) {:.4f}'.format(t, epochs_processed, progress_percent * 100, print_loss_avg))
                
            print(f'Epoch {i}, loss_avg: {loss_avg}')
            loss_avg /= float(len(train_paragraphs))
            print(f'\tAfter division: {loss_avg}')

            # Save loss value
            loss_avgs.append((i, loss_avg))
            with open(LOSS_FILE_FORMAT.format(self.log.dir), 'a+') as f:
                f.write('{},{}\t'.format(i, loss_avg))

            # Calculate loss on validation set:
            if i > 0 and i % validate_every == 0:

                validation_loss_avg = 0
                for j, validation_paragraph in enumerate(validation_paragraphs):
                    # Create copies
                    encoder_optimizer_copy = copy.deepcopy(self.encoder_optimizer)
                    decoder_optimizer_copy = copy.deepcopy(self.decoder_optimizer)
                    context_optimizer_copy = copy.deepcopy(self.context_optimizer)
                    encoder_copy = copy.deepcopy(self.encoder)
                    decoder_copy = copy.deepcopy(self.decoder)
                    context_copy = copy.deepcopy(self.context)
                    context_hidden_copy = copy.deepcopy(self.context_hidden)
                    criterion_copy = copy.deepcopy(criterion)
                    
                    loss, _ = self._train_paragraph(validation_paragraph,
                            encoder_copy, decoder_copy, context_copy,
                            context_hidden_copy, encoder_optimizer_copy, decoder_optimizer_copy, context_optimizer_copy, 
                            criterion_copy)
                    validation_loss_avg += loss
                # Save validation loss value
                validation_loss_avg /= float(len(validation_paragraphs))
                validation_loss_avgs.append((i, validation_loss_avg))
                with open(VALIDATION_LOSS_FILE_FORMAT.format(self.log.dir), 'a+') as f:
                    f.write('{},{}\t'.format(i, validation_loss_avg))
            
            # Save a checkpoint
            if save_temp_models:
                checkpoint_filename = CHECKPOINT_FILE_FORMAT.format(
                    CHECKPOINT_DIR, i, self.embedding_size, self.hidden_size, self.max_length, self.optimizer_type)
                checkpoint_file = Path(checkpoint_filename)
                # Save model at current epoch if doesn't exist
                if not checkpoint_file.is_file():
                    self.log.debug(logfile, 'Saving temporary model at epoch={}'.format(i))
                    self.saveToFiles(checkpoint_file)
                # Delete second previous model if not a multiple of 10
                if i > 0 and ((i-1) % checkpoint_every != 0 or (i-1) == 0):
                    # Delete model with epoch = i-1
                    checkpoint_file = Path(CHECKPOINT_FILE_FORMAT.format(
                        CHECKPOINT_DIR, i-1, self.embedding_size, self.hidden_size, self.max_length, self.optimizer_type))
                    if checkpoint_file.is_file():
                        checkpoint_file.unlink()
                        self.log.debug(logfile, 'Deleted temporary model at epoch={}'.format(i-1))
                    else:
                        self.log.error(logfile, 'Could not find temporary model at epoch={}'.format(i-1))
            
            
        """
        # Save the entirety of the loss values
        # TODO: re-add for above additions
        """
        loss_logfile = self.log.create('train_model-loss_values')
        self.log.debug(loss_logfile, 'Validation loss averages: {}'.format(validation_loss_avgs))
        self.log.debug(loss_logfile, 'Loss averages: {}'.format(loss_avgs))
        self.log.info(logfile, 'Finished training on data for {} epochs.'.format(epochs))
        self.log.debug(logfile, 'Average loss={:.4f}'.format(float(sum([item[1] for item in loss_avgs]))/len(loss_avgs)))
        if len(validation_loss_avgs) > 0:
            self.log.debug(logfile, 'Average validation loss={:.4f}'.format(float(sum([item[1] for item in validation_loss_avgs])) / len(validation_loss_avgs)))
        plt.plot([item[0] for item in loss_avgs], [item[1] for item in loss_avgs], label='Training')
        plt.plot([item[0] for item in validation_loss_avgs], [item[1] for item in validation_loss_avgs], label='Validation')
        
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('{}loss_figure.png'.format(self.log.dir)) # Save plot to log folder
        

    ######################################################################
    # Evaluation
    # ==========
    #
    # Evaluation is mostly the same as training, but there are no targets so
    # we simply feed the decoder's predictions back to itself for each step.
    # Every time it predicts a word we add it to the output string, and if it
    # predicts the EOL token we stop there. We also store the decoder's
    # attention outputs for display later.
    #
    def _evaluate(self, paragraph):
        with torch.no_grad():
            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)
            context_hidden = self.context.initHidden(self.device)

            context_outputs = torch.zeros(self.max_context, self.context_hidden_size, device=self.device)

            tensors = tensorsFromParagraph(self.book, paragraph, self.device)
            
            for i in range(len(tensors)-1):
                last = False
                if i+1 == len(tensors)-1:
                    last = True
                input_variable = tensors[i]
                input_length = input_variable.size()[0]
                encoder_hidden = self.encoder.initHidden(self.device)

                encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

                for ei in range(input_length):
                    if ei < self.max_length:
                        encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
                        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

                # calculate context
                context_output, context_hidden = self.context(encoder_hidden, context_hidden)
                context_outputs[i] = context_hidden

                # Do we need to care that we don't decode the first pairs?
                if last:
                    target_variable = tensors[i+1]

                    decoder_input = torch.tensor([[START_ID]], device=self.device)  # SOS

                    decoder_hidden = None

                    decoder_inputs = [decoder_input]
                    decoder_hiddens = [decoder_hidden]
                    for di in range(self.max_length):
                        decoder_output, decoder_hidden, decoder_attention = self.decoder(
                            decoder_input, decoder_hidden, encoder_outputs, context_hidden, context_outputs)
                        
                        topv, topi = decoder_output.data.topk(1)

                        if topi.item() == book.STOP_ID:
                            decoded_words.append(book.STOP_TOKEN)
                            break
                        else:
                            decoded_words.append(self.book.index2word[topi.item()])

                        decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]
    
    # NOTE: original functions
    # Performs beam search on the data to find better scoring sentences (evaluate uses a "beam search" of k=1)
    def beam_search(self, paragraph, k):
        with torch.no_grad():
            tensors = tensorsFromParagraph(self.book, paragraph, self.device)
                
            for i in range(len(tensors)-1):
                last = False
                if i+1 == len(tensors)-1:
                    last = True
                input_variable = tensors[i]
                input_length = input_variable.size()[0]
                encoder_hidden = self.encoder.initHidden(self.device)

                encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

                for ei in range(input_length):
                    if ei < self.max_length:
                        encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
                        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

                # calculate context
                context_output, context_hidden = self.context(context_hidden, encoder_hidden)

                # Do we need to care that we don't decode the first pairs?
                if last:
                    target_variable = tensors[i+1]

                    decoder_input = torch.tensor([[START_ID]], device=self.device)  # SOS

                    decoder_hidden = context_output

                    # Get initial k results:
                    decoder_output, decoder_hidden, _ = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs, context_hidden)
                    topv, topi = decoder_output.data.topk(k)
                    results = []
                    for i in range(k):
                        # Score, item tensor, hidden layer
                        results.append(BeamSearchResult(topv.squeeze()[i].item(), topi.squeeze()[i].detach(), decoder_hidden))
                    ###
                    # Expand the search for topk for each result until we have 5 sentences:
                    sentence_length = 0
                    while sentence_length <= self.max_length:
                        new_results = [] # We will have k*k results in this after for-loop, then sort and take best k
                        still_searching = False
                        for result in results:
                            if not result.stopped:
                                still_searching = True
                                decoder_output, decoder_hidden, _ = self.decoder(
                                    result.get_latest_item(), result.hidden, encoder_outputs, context_hidden)
                                topv, topi = decoder_output.data.topk(k)
                                for i in range(k):
                                    new_result = result.add_result(BeamSearchResult(topv.squeeze()[i].item(), topi.squeeze()[i].detach(), decoder_hidden))
                                    # If the next generated word is EOL, stop the sentence
                                    if topi.squeeze()[i].item() == STOP_ID:
                                        new_result.stopped = True
                                    else:
                                        new_result.words.append(self.book.index2word[topi.squeeze()[i].item()])
                                    new_results.append(new_result)
                            else: # make sure to re-add currently stopped sentences
                                new_results.append(result)
                        results = sorted(new_results, key=operator.attrgetter('score'))[::-1][:k]
                        if not still_searching:
                            break
                        # Prevent beam_search from being stuck in an infinite loop
                        sentence_length += 1
                    ###
                    
                    return results    

    #NOTE: original functions
    # Forces the model to generate the 'sentence_to_evaluate' and records its perplexity per word
    
    def _evaluate_specified(self, paragraph, last_sentence):
        with torch.no_grad():
            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)
            context_hidden = self.context.initHidden(self.device)

            tensors = tensorsFromParagraph(self.book, paragraph, self.device)

            for i in range(len(tensors)-1):
                last = False
                if i+1 == len(tensors)-1:
                    last = True
                input_variable = tensors[i]
                input_length = input_variable.size()[0]
                encoder_hidden = self.encoder.initHidden(self.device)

                encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

                for ei in range(input_length):
                    if ei < self.max_length:
                        encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden)
                        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

                # calculate context
                context_output, context_hidden = self.context(context_hidden, encoder_hidden)


                # Do we need to care that we don't decode the first pairs?
                if last:
                    target_variable = tensors[i+1]

                    decoder_input = torch.tensor([[START_ID]], device=self.device)  # SOS

                    decoder_hidden = context_output

                    decoded_words = []
                    decoder_attentions = torch.zeros(self.max_length, self.max_length)

                    summation = 0.0
                    N = 0

                    evaluate_variable = tensorFromSentence(self.book, last_sentence, self.device)

                    # Previous error here: TypeError: iteration over a 0-d tensor
                    try:
                        evaluate_items = [t.item() for t in evaluate_variable.squeeze()]
                    except TypeError as e:
                        print('TypeError thrown: {}'.format(repr(e)))
                        return None

                    for evaluate_item in evaluate_items:
                        N += 1

                        decoder_output, decoder_hidden, decoder_attention = self.decoder(
                            decoder_input, decoder_hidden, encoder_outputs, context_hidden)
                        
                        topv, topi = decoder_output.data.topk(1)

                        ## Perplexity code ##
                        # We need to get the value for the item we're evaluating, not what was predicted:
                        # decoder_output.data is of form tensor([[x, y, ..., z]]) where each value is a log value
                        # The index in this tensor is the index of the word in the book
                        summation += decoder_output.data.squeeze()[evaluate_item].item()
                    
                        if evaluate_item == STOP_ID:
                            break
                        else:
                            # Decode the predicted word from the book 
                            decoded_words.append(self.book.index2word[topi.item()])

                        decoder_input = torch.tensor([[evaluate_item]], device=self.device)

                    perplexity = pow(math.e, -summation / N)# / N because the evaluate sentence is converted to a tensor where the last item will be STOP_ID

                    # note: decoder_attentions not properly set up in this function

                    #return decoded_words, decoder_attentions, perplexity
                    return perplexity

    #######
    # TODO: Implement evaluate_test_set