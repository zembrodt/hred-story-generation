# perplexity_study.py
import getopt
import math
import matplotlib.pyplot as plt
from pathlib import Path
import os, pickle, random, re, sys

import hred_story_generation as storygen
from storygen.hred import Hred
import storygen.log as log

OBJ_DIR = 'obj'
EMBEDDING_TYPES = ['glove', 'sg', 'cbow']
HELP_MSG = ''.join([
    'Usage:\n',
    'python3 perplexity_study.py [-h, --help] [--embedding]\n',
    '\t-h, --help: Provides help on command line parameters\n',
	'\t--embedding <embedding_type>: specify an embedding to use from: {}'.format(EMBEDDING_TYPES),
    '\t--build <filename>: builds a graphical representation of several perplexity studies',
])


def calculate_perplexities(network, words, paragraphs, sentences, sentences_by_lengths):
     # Actual book sentences
    actual_sentences_score = 0.0

    # Random words
    random_words_score = 0.0
    
    # Random sentences
    random_sentences_score = 0.0
    
    # Exact random sentences
    exact_random_sentences_score = 0.0

    # Calculate scores
    i = 0
    percentages = {0}
    curr_len = len(paragraphs)
    for paragraph in paragraphs:
        # Calculate the actual perplexity
        actual_perplexity = network._evaluate_specified(paragraph, paragraph[-1])

        # Calculate the perplexity of sentences (of the same length) build of random words in the vocab 
        random_words = [random.choice(words) for i in range(len(paragraph[-1]))]
        words_perplexity = network._evaluate_specified(paragraph, random_words)
        
        # Calculate the perplexity of sentences taken at random from the pairs
        random_sentence = random.choice(sentences)
        sentence_perplexity = network._evaluate_specified(paragraph, random_sentence)

        # Calculate the perplexity of sentences taken at random of the same length as target sentence
        # TODO: pre-calculate a dictionary for all sentence lengths
        exact_random_sentence = random.choice(sentences_by_lengths[len(paragraph[-1])])
        exact_sentence_perplexity = network._evaluate_specified(paragraph, exact_random_sentence)

        if actual_perplexity is not None and words_perplexity is not None and sentence_perplexity is not None and exact_sentence_perplexity is not None:
            actual_sentences_score += actual_perplexity
            random_words_score += words_perplexity
            random_sentences_score += sentence_perplexity
            exact_random_sentences_score += exact_sentence_perplexity
        else:
            curr_len -= 1
            if actual_perplexity is None:
                print(f'(i={i}) Actual: retrieved a 0-d tensor from ({paragraph})')
            if words_perplexity is None:
                print(f'(i={i}) Random words: retrieved a 0-d tensor from ([{paragraph}], [{random_words}])')
            if sentence_perplexity is None:
                print(f'(i={i}) Random sentences: retrieved a 0-d tensor from ([{paragraph}], [{random_sentence}])')
            if exact_sentence_perplexity is None:
                print(f'(i={i}) Exact random sentences: retrieved a 0-d tensor from ([{paragraph}], [{exact_random_sentence}])')

        i += 1
        percentage = math.floor(100 * (float(i) / len(paragraphs)))
        if percentage % 5 == 0 and percentage not in percentages:
            print('{}% complete.'.format(percentage))
            percentages.add(percentage)

    return actual_sentences_score / curr_len, random_words_score / curr_len, random_sentences_score / curr_len, exact_random_sentences_score / curr_len

# Study on the perplexity module using the provided text
def main(argv):
    # Get command line arguments
    try:
        opts, _ = getopt.getopt(argv, 'hf:', ['embedding=', 'build=', 'help', 'file='])
    except getopt.GetoptError as e:
        print(e)
        print(HELP_MSG)
        exit(2)

    # Logger
    logger = log.Log()
    logfile = logger.create('perplexity_study')

    # Default value
    embedding_type = None
    checkpoint_filename = None

    # Set values from command line
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(HELP_MSG)
            exit()
        if opt in ('-f', '--file'):
            checkpoint_filename = arg
        elif opt == '--embedding':
            embedding_type = arg
            if embedding_type not in EMBEDDING_TYPES:
                print('{} is not a valid embedding type'.format(embedding_type))
                print(HELP_MSG)
                exit(2)
        elif opt == '--build':
            build_file = Path(arg)
            if build_file.is_file():
                x_labels = ['Actual score', 'Random words', 'Random sentences']
                with open(arg, 'r') as f:
                    for i, line in enumerate(f.readlines()):
                        #embedding_type,epochs,actual_score_train,actual_score_test,random_words_train,random_words_test,random_sentences_train,random_sentences_test
                        embedding_type, epochs, actual_scores, random_words_scores, random_sentences_scores = line.split(';')
                        actual_score_train, actual_score_test = actual_scores.split(',')
                        random_words_train, random_words_test = random_words_scores.split(',')
                        random_sentences_train, random_sentences_test = random_sentences_scores.split(',')
                        label_train = '{}_{}_train'.format(embedding_type, epochs)
                        label_test = '{}_{}_test'.format(embedding_type, epochs)
                        plt.plot(x_labels, [actual_score_train, random_words_train, random_sentences_train], label=label_train)
                        plt.plot(x_labels, [actual_score_test, random_words_test, random_sentences_test], label=label_test)
                plt.legend()
                plt.xlabel('Score type')
                plt.xlabel('Perplexity')
                plt.show()
                exit(0)
            else:
                print('{} is not a file'.format(arg))
                exit(1)

    print('Embedding type = {}'.format(embedding_type))
    logger.info(logfile, 'Embedding type = {}'.format(embedding_type))
    # Get directory name for our embedding type
    obj_dir = OBJ_DIR
    if embedding_type is not None:
        obj_dir += '_{}/'.format(embedding_type)
    else:
        obj_dir += '/'

    # prepare data
    train_paragraphs, validation_paragraphs, test_paragraphs = [], [], []
    with open('data/train_raw.pkl', 'rb') as f:
        train_paragraphs = pickle.load(f)
    with open('data/validate_raw.pkl', 'rb') as f:
        validation_paragraphs = pickle.load(f)
    with open('data/test_raw.pkl', 'rb') as f:
        test_paragraphs = pickle.load(f)

    paragraphs = train_paragraphs + validation_paragraphs + test_paragraphs

    # Create a list of all sentences
    test_sentences = [sentence for paragraph in test_paragraphs for sentence in paragraph]
    test_sentences_by_lengths = {}
    for sentence in test_sentences:
        if len(sentence) in test_sentences_by_lengths:
            test_sentences_by_lengths[len(sentence)].append(sentence)
        else:
            test_sentences_by_lengths[len(sentence)] = [sentence]

    MAX_LENGTH = max(
        max(map(len, [sentence for sentence in paragraph]))
    for paragraph in paragraphs)
    MAX_LENGTH += 1 # for <EOL> token

    book_title = '1_sorcerers_stone'

    # Create a book object from the train/test pairs
    book = storygen.get_book(book_title, paragraphs)    
    
    # Create and load the model
    hred = Hred(
        hidden_size=storygen.HIDDEN_SIZE,
        max_length=MAX_LENGTH,
        embedding_size=storygen.EMBEDDING_SIZE,
        optimizer_type='sgd', # Currently hard-coded for SGD optimizers
        book=book,
        use_cuda=storygen.USE_CUDA
    )
    if not hred.loadFromFiles(checkpoint_filename):
        print(f'Error loading checkpoint at {checkpoint_filename}')
        logger.error(logfile, f'Error loading checkpoint at {checkpoint_filename}')
        exit(1)

    EPOCHS = 100
    words = list(book.word2index)

    # Final scores
    final_actual_sentences_score_test = 0.0
    final_random_words_score_test = 0.0
    final_random_sentences_score_test = 0.0
    final_exact_random_sentences_score_test = 0.0

    for i in range(EPOCHS):
        print(f'Processing epoch {i+1}...')
        # Actual book sentences
        #actual_sentences_score_train = 0.0
        actual_sentences_score_test = 0.0

        # Random words
        #random_words_score_train = 0.0
        random_words_score_test = 0.0
        #words = list(book.word2index)
        
        # Random sentences
        #random_sentences_score_train = 0.0
        random_sentences_score_test = 0.0

        # Excat random sentences
        #exact_random_sentences_score_train = 0.0
        exact_random_sentences_score_test = 0.0
        
        # Calculate scores on training data
        #print('Calculating scores on training data...')

        #actual_sentences_score_train, random_words_score_train, random_sentences_score_train, exact_random_sentences_score_train = calculate_perplexities(hred, words, train_paragraphs)

        #print('Calculating scores on test data...')
        actual_sentences_score_test, random_words_score_test, random_sentences_score_test, exact_random_sentences_score_test = calculate_perplexities(hred, words, test_paragraphs, test_sentences, test_sentences_by_lengths)
        
        logger.info(logfile, f'Processing epoch {i+1}')

        #print('Actual sentences score (train): {:.4f}'.format(actual_sentences_score_train))
        #print('Actual sentences score (test): {:.4f}'.format(actual_sentences_score_test))
        #logger.info(logfile, 'Actual sentences score (train): {:.4f}'.format(actual_sentences_score_train))
        logger.info(logfile, 'Actual sentences score (test): {:.4f}'.format(actual_sentences_score_test))

        #print('Random words score (train): {:.4f}'.format(random_words_score_train))
        #print('Random words score (test): {:.4f}'.format(random_words_score_test))
        #logger.info(logfile, 'Random words score (train): {:.4f}'.format(random_words_score_train))
        logger.info(logfile, 'Random words score (test): {:.4f}'.format(random_words_score_test))

        #print('Random sentences score (train): {:.4f}'.format(random_sentences_score_train))
        #print('Random sentences score (test): {:.4f}'.format(random_sentences_score_test))
        #logger.info(logfile, 'Random sentences score (train): {:.4f}'.format(random_sentences_score_train))
        logger.info(logfile, 'Random sentences score (test): {:.4f}'.format(random_sentences_score_test))

        #print('Exact random sentences score (train): {:.4f}'.format(exact_random_sentences_score_train))
        #print('Exact random sentences score (test): {:.4f}'.format(exact_random_sentences_score_test))
        #logger.info(logfile, 'Exact random sentences score (train): {:.4f}'.format(exact_random_sentences_score_train))
        logger.info(logfile, 'Exact random sentences score (test): {:.4f}'.format(exact_random_sentences_score_test))

        final_actual_sentences_score_test += actual_sentences_score_test
        final_random_words_score_test += random_words_score_test
        final_random_sentences_score_test += random_sentences_score_test
        final_exact_random_sentences_score_test += exact_random_sentences_score_test

    final_actual_sentences_score_test /= float(EPOCHS)
    final_random_words_score_test /= float(EPOCHS)
    final_random_sentences_score_test /= float(EPOCHS)
    final_exact_random_sentences_score_test /= float(EPOCHS)

    logger.info(logfile, 'Final perplexity scores:')

    print('Actual sentences score (test): {:.4f}'.format(final_actual_sentences_score_test))
    logger.info(logfile, 'Actual sentences score (test): {:.4f}'.format(final_actual_sentences_score_test))

    print('Random words score (test): {:.4f}'.format(final_random_words_score_test))
    logger.info(logfile, 'Random words score (test): {:.4f}'.format(final_random_words_score_test))

    print('Random sentences score (test): {:.4f}'.format(final_random_sentences_score_test))
    logger.info(logfile, 'Random sentences score (test): {:.4f}'.format(final_random_sentences_score_test))

    print('Exact random sentences score (test): {:.4f}'.format(final_exact_random_sentences_score_test))
    logger.info(logfile, 'Exact random sentences score (test): {:.4f}'.format(final_exact_random_sentences_score_test))

if __name__ == '__main__':
    main(sys.argv[1:])
