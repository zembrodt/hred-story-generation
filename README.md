# hred-story-generation

Story generation project using a Hierarchical Recurrent Encoder-Decoder

The HRED architecture was implemented for story generation in order to compare with a [previously implemented](https://github.com/zembrodt/story-generation) sequence-to-sequence network.
An HRED takes this previous implementation a step further: it uses the context of an entire paragraph when predicting each sentence. This is done with the belief that it is difficult for a model to learn to predict sentences solely based on the previous one. Stories make reference and use of previous information information, and this project explores the value of bringing the context for prediction to a paragraph-level.  
The implementation was built off the previous sequence-to-sequence one by implementing a context-encoder RNN and modifying the decoder RNN to use this context.

## Requirements

-   [Python 3.x](https://www.python.org/downloads/)
-   [pytorch](https://pytorch.org/)
-   [nltk](https://www.nltk.org/)
-   [pymeteor](https://github.com/zembrodt/pymeteor)

**_Note_**: to install pymeteor you must use the Test PyPi server:<br/>
`pip install --index-url https://test.pypi.org/simple/ pymeteor`

Tested with Python 3.6.6, PyTorch 0.4.1, and Cuda 9.0.176

Download the Wikipedia 2014 + Gigaword 5 pre-trained vectors from [GloVe](https://nlp.stanford.edu/projects/glove/), download link [here](http://nlp.stanford.edu/data/glove.6B.zip).<br/>
Unzip the text files to the location `data/glove.6B/`

### Usage

`python3 hred_story_generation.py` or `./hred_story_generation.py`<br/>
All command line arguments are optional, and any combination (beides `-h, --help`) can be used.<br/>
Arguments:

-   `-h, --help` : Provides help on command line parameters
-   `--epoch <epoch_value>` : specify an epoch value to train the model for or load a checkpoint from
-   `--embedding <embedding_type>` : specify an embedding to use from: `[glove, sg, cbow]`
-   `--optim, --optimizer <optimizer_type>` : specify the type of optimizer to use from: `[adam, sgd]`
-   `--loss <loss_dir>` : specify a directory to load loss values from (requires files `loss.dat` and `validation.dat`)
-   `--largedata` : specifies to use a large dataset for training/testing (four books instead of one)'])

Along with `hred_story_generation.py`, several other files can be executed as standalone scripts:

-   `perplexity_study.py` allows the user to gather perplexity results from a specified model's checkpoint using the `-f` or `--file` parameter
-   `storygen/book.py` provides use to parse or filter standalone text into new files. `./book.py -h` for more information
-   `util/display_loss.py` allows the user to display the loss values for select word embeddings, with or without validation values. `./display_loss.py -h` for more information
-   `util/loss_analysis.py` allows the user to view min/max loss values of a given file, or find loss values at a specific epoch. `./loss_analysis.py -h` for more information

## Current Results

Three models were trained for each embedding:

-   Random
-   Pre-trained _GloVe_
-   Custom _word2vec_ CBOW

Each model was trained with a learning rate of 0.0001 and with SGD optimization for 1000 epochs on 166 paragraphs, with another 42 paragraphs used for validation. The final loss value for random, _GloVe_, and _word2vec_ CBOW were **3.822**, **4.07**, and **3.867** respectively. These loss values are around 0.8 to 1 higher than the values achieved by the sequence-to-sequence models, but this can be attributed to a much smaller dataset. Their full training and validation loss value graphs can be found below:<br/>
<img src="https://i.imgur.com/Eloqe7Q.png" alt="Model trained with random embedding" /><br/>
<img src="https://i.imgur.com/djdNXAo.png" alt="Model trained with pre-trained *GloVe* embedding" /><br/>
<img src="https://i.imgur.com/uRqFkmS.png" alt="Model trained with custom *word2vec* CBOW embedding" /><br/>

A perplexity study was ran on all three of these models. Their scores on the test dataset can be found in the table below:<br/>
| Embedding | | Perplexity |
| --------------- | ------------------------ | --------------- |
| Random | _Actual sentences_ | **2826.049** |
| | _Random sentences_ | 3125.4459 |
| | _Exact random sentences_ | **2812.2747** |
| | _Random words_ | 23585.7831 |
| _GloVe_ | _Actual sentences_ | **1174.725** |
| | _Random sentences_ | **1114.8894** |
| | _Exact random sentences_ | 1033.2969 |
| | _Random words_ | 18572.4679 |
| _word2vec_ CBOW | _Actual sentences_ | 1592.9276 |
| | _Random sentences_ | 1447.1607 |
| | _Exact random sentences_ | **1361.7506** |
| | _Random words_ | 24280.3667 |

Here, <em>actual sentences</em> refers to the score of the model being forced to evaluate the actual target sentence when given the previous sentences from the paragraph. <em>Random sentences</em> is forcing the model to evaluate a random target sentence taken from the test dataset. <em>Exact random sentences</em> is similar, but refers to forcing the model to evaluate a random target sentence that contains the same amount of tokens as the actual target sentence. Lastly, <em>random words</em> is forming a sentence from random words in the vocabulary that is of the same length as the actual target sentence, and forcing the model to evaluate it.

Out of the four categories of studies, we want <em>actual sentences</em> should perform the best, as they are the real sentences to follow the input sentences. This will be followed by <em>random sentences</em> or <em>exact random sentences</em>, as these are still actual sentences, and the model should be able to recognize that. The random sentence evaluation was split into these two to explore if sentence length was significant to a model's prediction. Ideally, these should still perform worst than <em>actual sentences</em>, as they will potentially have no context with the rest of the paragraph. Finally, <em>random words</em> should perform the worse, as they will most likely not have a real sentence structure.

The above table shows that all three sentence tests performed similarly, and were significantly smaller than <em>random words</em>. However, the two random sentence tests outperformed <em>actual sentences</em> in both custom embeddings, and <em>exact random sentences</em> outperformed <em>actuals</em> in the random embedding.

### info below out of date, updating soon

Finally, the actual perplexity themselves were extremely high for the testing data, further showing that perhaps the model hasn't learned enough to handle unseen data. One cause of this could be due to the lack for training data, with just over 5,000 sentence pairs.

## Future work

Beyond correcting current drawbacks, such as checkpoint loading issues, high perplexity values, and a minimum loss value of 3, future work could include:

-   Training and testing a working model on corpora of different types, such as news articles or song lyrics
-   Training more custom embeddings, either current ones for much longer, or using GloVe to train custom word embeddings rather than word2vec

## Previous Results

### 11/15/2018

Trained three word2vec embeddings on all Harry Potter texts: Skip-Gram and Continuous Bag of Words trained for 15 epochs, and Continuous Bag of Words trained for 300 epochs.<br/>
With these 3 word2vec embeddings, the previous GloVe embedding, and the default random embedding, trained five models for 500 epochs on the data.<br/>
The models still seem to be underfitting, with the word2vec embeddings outperforming the random embedding. GloVe embedding still performs the best. See the results for loss values in the figure below:<br/>
<img src="https://i.imgur.com/YZZjo1f.png" alt="Loss graph, 500 epochs, for random, GloVe, word2vec-sg-15, word2vec-cbow-15, word2vec-cbow-300" />

### 11/9/2018

Currently getting a minimum loss value of **2.977** until the loss spikes around the 30th epoch, as you can see in the figure below:<br/>
<img src="https://i.imgur.com/QyFyXIT.png" alt="Loss graph, 100 epochs" />

This seems to be due to the model beginning with the values from the above word embeddings, then breaking out and not being able to find the local optimum for the Harry Potter texts. An idea to correct this is to train our own word2vec on the Harry Potter texts.<br />
The model is also underfitting (on the 100th epoch) when evaluated, predicting sentences with repeated words.<br/>
Evaluating this model with beam search (k=1), the average perplexity is **100,562.7942**.<br/>
Evaluating this model with beam search (k=5), the average perplexity is **93,277.5684**.

Retraining this model on 40 epochs, we get an minimum loss value of **2.959** at the 23rd epoch.<br/>
Evaluating this model at k=1 gives us a perplexity value of **675.8603**.<br/>
Evaluating this model at k=5 gives us a perplexity value of **669.7439**.<br/>
Viewing prediction results at this point in training the model, it is apparent that the model is not yet underfitting.<br/>
Refer to the figure below to see the chosen minimum loss value at epoch=22, before the loss value spikes at epoch=30.<br/>
<img src="https://i.imgur.com/NhScaLG.png" alt="Loss graph, 40 epochs" />
