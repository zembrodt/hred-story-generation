# evaluate_model.py
import getopt
import math
import matplotlib.pyplot as plt
from pathlib import Path
import os, pickle, random, re, sys

import hred_story_generation as storygen
from storygen.hred import Hred
import storygen.log as log

HELP_MSG = ''.join([
    'Usage:\n',
    'python3 perplexity_study.py [-h, --help] [-f, --file <checkpoint file>]\n',
    '\t-h, --help: Provides help on command line parameters\n',
    '\t-f, --file <checkpoint file>: Specify the checkpoint file to load the model from\n',
])


# Evaluate a model on test data
def main(argv):
    # Get command line arguments
    try:
        opts, _ = getopt.getopt(argv, 'hf:', ['help', 'file=', 'largedata'])
    except getopt.GetoptError as e:
        print(e)
        print(HELP_MSG)
        exit(2)

    # Logger
    logger = log.Log()
    logfile = logger.create('evaluate_model')

    # Default value
    checkpoint_filename = None
    large_data = False

    # Set values from command line
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(HELP_MSG)
            exit()
        elif opt in ('-f', '--file'):
            checkpoint_filename = arg
        elif opt == '--largedata':
            large_data = True
        
    logger.info(logfile, f'Evaluating model:     {checkpoint_filename}')
    logger.info(logfile, f'Using large test set: {large_data}')

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
    book = storygen.get_book(book_title, paragraphs)    
    
    # Create and load the model
    hred = Hred(
        hidden_size=storygen.HIDDEN_SIZE,
        context_hidden_size=storygen.CONTEXT_HIDDEN_SIZE,
        max_length=MAX_LENGTH,
        max_context=storygen.MAX_CONTEXT,
        embedding_size=storygen.EMBEDDING_SIZE,
        optimizer_type='sgd', # Currently hard-coded for SGD optimizers
        book=book,
        device=storygen.DEVICE
    )
    if not hred.loadFromFiles(checkpoint_filename):
        print(f'Error loading checkpoint at {checkpoint_filename}')
        logger.error(logfile, f'Error loading checkpoint at {checkpoint_filename}')
        exit(1)

    print(f'Evaluating {len(test_paragraphs)} paragraphs')
    logger.info(logfile, f'Evaluating {len(test_paragraphs)} paragraphs')
    evaluate_train_every = 15
    for i, test_paragraph in enumerate(test_paragraphs):
        decoded_words, _ = hred._evaluate(test_paragraph)
        for sentence in test_paragraph[:-1]:
            logger.info(logfile, f'> {" ".join(sentence)}')
        logger.info(logfile, f'= {" ".join(test_paragraph[-1])}')
        logger.info(logfile, f'< {" ".join(decoded_words)}')

        if i % evaluate_train_every == 0:
            # Evaluate a train paragraph
            train_paragraph = random.choice(train_paragraphs)
            decoded_words, _ = hred._evaluate(train_paragraph)
            logger.info(logfile, '--- TRAIN DATA POINT ---')
            for sentence in train_paragraph[:-1]:
                logger.info(logfile, f'> {" ".join(sentence)}')
            logger.info(logfile, f'= {" ".join(train_paragraph[-1])}')
            logger.info(logfile, f'< {" ".join(decoded_words)}')

if __name__=='__main__':
    main(sys.argv[1:])