from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
import matplotlib.pyplot as plt
from tensorflow import keras
from nltk.corpus import stopwords
import nltk
import gensim.downloader
import numpy as np


def add_words_from_songs(songs_lyrics, word_to_index):
    """
    Add words from all songs to the vocabulary
    :param songs_lyrics: All songs lyrics
    :param word_to_index: Word to index dict
    :return: Updated word to index dict
    """
    all_words = set([w for s in songs_lyrics for w in s.split()])
    words_num = max(word_to_index.values())
    model_words = set(word_to_index.values())
    words_to_add = all_words.difference(model_words)
    dict_to_add = {w: i + words_num for i, w in enumerate(words_to_add)}
    word_to_index = {**word_to_index, **dict_to_add}

    return word_to_index


def tokenize(songs_lyrics, word_to_index):
    """ Tokenize the songs' lyrics
    :param songs_lyrics: All songs
    :param word_to_index: Word to index dict
    :return: Sequences of n-grams
    """
    nltk.download('stopwords')
    current_words = set(word_to_index.keys())
    swords = set(stopwords.words("english"))
    w_set = current_words.difference(swords)
    input_sequences = []

    for song in songs_lyrics:
        tokenized = tf.keras.preprocessing.text.text_to_word_sequence(
            song, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True, split=' ')

        tokenized = [word_to_index[w] for w in tokenized if w in w_set]
        for i in range(1, len(tokenized)):
            n_gram_sequence = tokenized[:i + 1]
            input_sequences.append(n_gram_sequence)

    return input_sequences


def create_dataset(csv_file_path: Path, index_to_word, word_to_index):
    """
    Generate the model dataset
    :param word_to_index:
    :param index_to_word:
    :param csv_file_path: A path to the csv file containing the artist name, song name and lyrics
    :return: Predictors and labels
    """
    df = pd.read_csv(csv_file_path, header=None)

    artists_names, songs_names, songs_lyrics = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    word_to_index = add_words_from_songs(songs_lyrics, word_to_index)
    tokenized_ngrams = tokenize(songs_lyrics, word_to_index)

    max_sequence_len = max([len(x) for x in tokenized_ngrams])
    tokenized_ngrams = np.array(pad_sequences(tokenized_ngrams, maxlen=max_sequence_len, padding='pre'))
    predictors, label = tokenized_ngrams[:, :-1], tokenized_ngrams[:, -1]

    return predictors, label


class RNN(keras.Model):

    def __init__(self, input_dim: int, pretrained_weights, output_dim):
        super(RNN, self).__init__()
        """
        input_dim = Vocab size
        output_dim = embedding size
        pretrained_weights = imported embedding matrix
        """
        self.embed1 = Embedding(input_dim=input_dim, output_dim=output_dim)
        self.embed1.build((None,))  # hack for setting the embedding matrix
        self.embed1.set_weights([pretrained_weights])
        self.embed1.trainable = False
        self.lstm1 = LSTM(units=256, return_sequences=True)
        self.lstm2 = LSTM(units=256, return_sequences=True)
        self.dense = Dense(units=input_dim, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        # input: [batch_size, input_length]
        x = self.embed1(inputs)  # [batch_size, timesteps, units]
        x = self.lstm1(x)  # [batch_size, timesteps, units]
        x = self.lstm2(x)  # [batch_size, timesteps, units]
        x = self.dense(x)  # [batch_size, timesteps, vocabulary_size]

        return x


def train(csv_path,
          BATCH_SIZE=32,
          epochs=100
          ):
    word2vec_model_name = 'glove-twitter-25'
    word2vec_model = gensim.downloader.load(word2vec_model_name)
    pretrained_weights = word2vec_model.vectors
    index_to_word = word2vec_model.index_to_key
    word_to_index = word2vec_model.key_to_index

    predictors, label = create_dataset(csv_path, index_to_word, word_to_index)

    """
    TODOLIST:
    Notes: 
        a) Do not remove the & 
    1) create the dataset s.t:
        each member of the dataset is : (x, y)
        x is 1-D list of shape = (timesteps) ,   (batched)====> (batch_size, timesteps) => (batch_size, timesteps, output_dim)
        y is shape = (timesteps, vocab_size) where we have 1 in the true word. (batched) ====> (batch_size, timesteps, vocab_size)
        
    2) debug to the current architecture of the model (call with mocked input)
    3) write the generation function
    4) debug the generation function with random model with our architecture
    5) feature engineering of midi
    6) add layer of adding the midi features to the model
    7) debug with mock
    8) train.
    """

    vocab_size, embedding_size = pretrained_weights.shape
    output_dim = int(word2vec_model_name.split('-')[-1])

    RNN_model = RNN(input_dim=vocab_size,
                    pretrained_weights=pretrained_weights,
                    output_dim=output_dim)
    RNN_model.compile(optimizer='adam',
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

    history = RNN_model.fit(predictors, label,
                            epochs=epochs,
                            verbose=1)

    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()


csv_filename = 'lyrics_train_set.csv'
csv_path = Path(rf'{csv_filename}')
train(csv_path)
