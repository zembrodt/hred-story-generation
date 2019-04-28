import getopt, pickle, random, sys, torch
from storygen.hred import Hred, OPTIMIZER_TYPES
from storygen.book import Book
from storygen.glove import DIMENSION_SIZES
from storygen.log import Log

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hidden vector sizes taken from https://arxiv.org/abs/1507.02221
HIDDEN_SIZE = 1000
CONTEXT_HIDDEN_SIZE = 1500
MAX_CONTEXT = 5
EMBEDDING_SIZE = DIMENSION_SIZES[-1]
DATA_FILE_FORMAT = 'data/{}_{}_{}.txt'

EMBEDDINGS = ['glove', 'cbow', 'sg']

# Help message for command line arguments
# TODO: this may need to be updated
HELP_MSG = '\n'.join([
		'Usage:',
		'python3 hred_story_generation.py [-h, --help] [--epoch <epoch_value>] [--embedding <embedding_type>] [--loss <loss_dir>] [--optim, --optimizer <optimizer_type>], [--largedata]',
		'\tAll command line arguments are optional, and any combination (beides -h) can be used',
		'\t-h, --help: Provides help on command line parameters',
		'\t--epoch <epoch_value>: specify an epoch value to train the model for or load a checkpoint from',
		'\t--embedding <embedding_type>: specify an embedding to use from: [glove, cbow, sg]',
        '\t--optim, --optimizer <optimizer_type>: specify the type of optimizer to use from: [adam, sgd]',
		'\t--loss <loss_dir>: specify a directory to load loss values from (requires files loss.dat and validation.dat)',
        '\t--largedata: specifies to use a large dataset for training/testing (four books instead of one)'])


# Creates a book object from the given train/test pairs
def get_book(book_title, paragraphs):
    bk = Book(book_title)

    #pairs = train_pairs + test_pairs
    for paragraph in paragraphs:
        for sentence in paragraph:
            bk.addSentence(sentence)

    return bk

def main(argv):
    log = Log()
    logfile = log.create('hred-story-generation')

    # Get command line arguments
    try:
        opts, _ = getopt.getopt(argv, 'h', 
            ['epoch=', 'embedding=', 'optim=', 'optimizer=', 'loss=', 'largedata', 'ca', 'help', 'cdir='])
    except getopt.GetoptError as e:
        print(e)
        print(HELP_MSG)
        exit(2)

    # Default values
    epoch_size = 100
    embedding_type = None
    optimizer_type = 'sgd'
    loss_dir = None
    large_data = False
    use_context_attention = False
    checkpoint_dir = None

    # Set values from command line
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(HELP_MSG)
            exit()
        # How many epochs to train for
        elif opt == '--epoch':
            try:
                epoch_size = int(arg)
            except ValueError:
                print('{} is not an integer. Argument must be an int.'.format(arg))
                exit()
        # The type of embedding to use
        elif opt == '--embedding':
            embedding_type = arg
        # The type of optimizer to use
        elif opt in ('--optim', '--optimizer'):
            if arg not in OPTIMIZER_TYPES:
                print(f'{arg} is not a correct optimizer type. Types are: {OPTIMIZER_TYPES}')
                exit()
            else:
                optimizer_type = arg
        # Directory to load previous loss values from
        elif opt == '--loss':
            loss_dir = arg
        # Use the large set of data (4 books instead of 1)
        elif opt == '--largedata':
            large_data = True
        # Use context attention
        elif opt == '--ca':
            use_context_attention = True
        # Manually set the checkpoint directory
        elif opt == '--cdir':
            checkpoint_dir = arg

    print('Epoch size            = {}'.format(epoch_size))
    print('Embedding type        = {}'.format(embedding_type))
    print('Optimizer type        = {}'.format(optimizer_type))
    print('Loss directory        = {}'.format(loss_dir))
    print('Hidden layer size     = {}'.format(HIDDEN_SIZE))
    print('Context hidden size   = {}'.format(CONTEXT_HIDDEN_SIZE))
    print('Use large data        = {}'.format(large_data))
    print('Use context attention = {}'.format(use_context_attention))
    if checkpoint_dir is not None:
        print(f'Checkpoint dir        = {checkpoint_dir}')
    print(f'Log dir               = {log.dir}')

    log.info(logfile, f'Epoch size            = {epoch_size}')
    log.info(logfile, f'Embedding type        = {embedding_type}')
    log.info(logfile, f'Optimizer type        = {optimizer_type}')
    log.info(logfile, f'Loss directory        = {loss_dir}')
    log.info(logfile, f'Hidden layer size     = {HIDDEN_SIZE}')
    log.info(logfile, f'Context hidden size   = {CONTEXT_HIDDEN_SIZE}')
    log.info(logfile, f'Use large data        = {large_data}')
    log.info(logfile, f'Use context attention = {use_context_attention}')
    if checkpoint_dir is not None:
        log.info(logfile, f'Checkpoint dir        = {checkpoint_dir}')

    # prepare data
    train_paragraphs, validation_paragraphs, test_paragraphs = [], [], []
    if large_data:
        with open('data/train_raw_4.pkl', 'rb') as f:
            train_paragraphs = pickle.load(f)
        with open('data/validate_raw_4.pkl', 'rb') as f:
            validation_paragraphs = pickle.load(f)
        with open('data/test_raw_4.pkl', 'rb') as f:
            test_paragraphs = pickle.load(f)
    else:
        with open('data/train_raw.pkl', 'rb') as f:
            train_paragraphs = pickle.load(f)
        with open('data/validate_raw.pkl', 'rb') as f:
            validation_paragraphs = pickle.load(f)
        with open('data/test_raw.pkl', 'rb') as f:
            test_paragraphs = pickle.load(f)

    paragraphs = train_paragraphs + validation_paragraphs + test_paragraphs

    print(f'Train:      {len(train_paragraphs)}')
    print(f'Validation: {len(validation_paragraphs)}')
    print(f'Test:       {len(test_paragraphs)}')
    print(f'Total:      {len(paragraphs)}')
    log.info(logfile, f'Train:      {len(train_paragraphs)}')
    log.info(logfile, f'Validation: {len(validation_paragraphs)}')
    log.info(logfile, f'Test:       {len(test_paragraphs)}')
    log.info(logfile, f'Total:      {len(paragraphs)}')

    MAX_LENGTH = max(
        max(map(len, [sentence for sentence in paragraph]))
    for paragraph in paragraphs)
    MAX_LENGTH += 1 # for <EOL> token

    book_title = '1_sorcerers_stone'

    # Create a book object from the train/test pairs
    book = get_book(book_title, paragraphs)    

    print('Creating HRED')
    hred = Hred(DEVICE, book, 
            MAX_LENGTH, MAX_CONTEXT, HIDDEN_SIZE, CONTEXT_HIDDEN_SIZE, 
            EMBEDDING_SIZE, optimizer_type,
            use_context_attention=use_context_attention,
            checkpoint_dir=checkpoint_dir
    )

    print(f'Training for {epoch_size} epochs')
    hred.train_model(epoch_size, train_paragraphs, validation_paragraphs,
            embedding_type=embedding_type, loss_dir=loss_dir, save_temp_models=True,
            checkpoint_every=50)

    print('Training complete.')

    print(f'Evaluating {len(test_paragraphs)} paragraphs')
    evaluate_train_every = 15
    for i, test_paragraph in enumerate(test_paragraphs):
        decoded_words, _ = hred._evaluate(test_paragraph)
        for sentence in test_paragraph[:-1]:
            log.info(logfile, f'> {" ".join(sentence)}')
        log.info(logfile, f'= {" ".join(test_paragraph[-1])}')
        log.info(logfile, f'< {" ".join(decoded_words)}')

        if i % evaluate_train_every == 0:
            # Evaluate a train paragraph
            train_paragraph = random.choice(train_paragraphs)
            decoded_words, _ = hred._evaluate(train_paragraph)
            for sentence in train_paragraph[:-1]:
                log.info(logfile, f'> {" ".join(sentence)}')
            log.info(logfile, f'= {" ".join(train_paragraph[-1])}')
            log.info(logfile, f'< {" ".join(decoded_words)}')

if __name__=='__main__':
    main(sys.argv[1:])