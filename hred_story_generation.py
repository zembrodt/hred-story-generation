import pickle
from storygen.hred import Hred
from storygen.book import Book

HIDDEN_SIZE = 256

# embeddings not yet implemented
# TODO: ONLY TEMPORARY FOR NO EMBEDDINGS:
EMBEDDING_SIZE = HIDDEN_SIZE

# Creates a book object from the given train/test pairs
def get_book(book_title, paragraphs):
    bk = Book(book_title)

    #pairs = train_pairs + test_pairs
    for paragraph in paragraphs:
        for sentence in paragraph:
            bk.addSentence(sentence)

    return bk

if __name__=='__main__':
    # prepare data
    paragraphs = []
    with open('data/train_raw.pkl', 'rb') as f:
        paragraphs = pickle.load(f)

    MAX_LENGTH = max(
        max(map(len, [sentence for sentence in paragraph]))
    for paragraph in paragraphs)
    MAX_LENGTH += 1 # for <EOL> token

    book_title = '1_sorcerers_stone'

    # Create a book object from the train/test pairs
    book = get_book(book_title, paragraphs)    

    print('Creating HRED')
    hred = Hred(groups=paragraphs,
            hidden_size=HIDDEN_SIZE,
            max_length=MAX_LENGTH,
            embedding_size=EMBEDDING_SIZE,
            book=book
    )
            #encoder_file='encoder_5.model',
            #decoder_file='decoder_5.model',
            #context_file='context_5.model')

    epochs = 10

    print(f'Training for {epochs} epochs')
    hred.train_model(epochs)