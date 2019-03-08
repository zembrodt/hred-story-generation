import getopt, pickle, random, sys
from storygen.hred import Hred, OPTIMIZER_TYPES
from storygen.book import Book
from storygen.glove import DIMENSION_SIZES
from storygen.log import Log

HIDDEN_SIZE = 256
EMBEDDING_SIZE = DIMENSION_SIZES[-1]
DATA_FILE_FORMAT = 'data/{}_{}_{}.txt'

EMBEDDINGS = ['glove', 'cbow', 'sg']

# Help message for command line arguments
# TODO: this may need to be updated
HELP_MSG = '\n'.join([
		'Usage:',
		'python3 story_generation.py [-h, --help] [--epoch <epoch_value>] [--embedding <embedding_type>] [--loss <loss_dir>]',
		'\tAll command line arguments are optional, and any combination (beides -h) can be used',
		'\t-h, --help: Provides help on command line parameters',
		'\t--epoch <epoch_value>: specify an epoch value to train the model for or load a checkpoint from',
		'\t--embedding <embedding_type>: specify an embedding to use from: [glove, cbow, sg]',
        '\t--optim <optimizer_type>: specity the type of optimizer to use from: [adam, sgd]',
		'\t--loss <loss_dir>: specify a directory to load loss values from (requires files loss.dat and validation.dat)',
		'\t-u, --update: specify that if previous train/test pairs exists, overwrite them with re-parsed data'])


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
        opts, _ = getopt.getopt(argv, 'hu', ['epoch=', 'embedding=', 'optim=', 'optimizer=', 'loss=', 'largedata', 'help', 'update'])
    except getopt.GetoptError as e:
        print(e)
        print(HELP_MSG)
        exit(2)

    # Default values
    epoch_size = 100
    embedding_type = None
    optimizer_type = 'adam'
    loss_dir = None
    load_previous = True
    large_data = False

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
        elif opt in ['--optim', '--optimizer']:
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
        # Flag to overwrite saved train/test pairs (if they exist)
        elif opt in ('-u', '--update'):
            load_previous = False

    print('Epoch size        = {}'.format(epoch_size))
    print('Embedding type    = {}'.format(embedding_type))
    print('Optimizer type    = {}'.format(optimizer_type))
    print('Loss directory    = {}'.format(loss_dir))
    print('Hidden layer size = {}'.format(HIDDEN_SIZE))

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

    MAX_LENGTH = max(
        max(map(len, [sentence for sentence in paragraph]))
    for paragraph in paragraphs)
    MAX_LENGTH += 1 # for <EOL> token

    book_title = '1_sorcerers_stone'

    # Create a book object from the train/test pairs
    book = get_book(book_title, paragraphs)    

    print('Creating HRED')
    hred = Hred(#groups=paragraphs,
            hidden_size=HIDDEN_SIZE,
            max_length=MAX_LENGTH,
            embedding_size=EMBEDDING_SIZE,
            optimizer_type=optimizer_type,
            book=book
    )
            #encoder_file='encoder_5.model',
            #decoder_file='decoder_5.model',
            #context_file='context_5.model')

    print(f'Training for {epoch_size} epochs')
    hred.train_model(epoch_size, train_paragraphs, validation_paragraphs,
            embedding_type=embedding_type, loss_dir=loss_dir, save_temp_models=True,
            checkpoint_every=5)

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