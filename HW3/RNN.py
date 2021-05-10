from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from nltk.corpus import stopwords
import nltk
import gensim.downloader
import numpy as np

csv_filename = 'lyrics_train_set.csv'


def tokenize(songs, word_to_index):
    nltk.download('stopwords')
    swords = set(stopwords.words("english"))

    input_sequences = []

    for song in songs:
        tokenized = tf.keras.preprocessing.text.text_to_word_sequence(
            song, filters=''''!"#$%&()*+,-.'/:;<=>?@[\\]^_`{|}~\t\n''',
            lower=True, split=' ')
        tokenized = [word_to_index[w] for w in tokenized if w not in swords]
        for i in range(1, len(tokenized)):
            n_gram_sequence = tokenized[:i + 1]
            input_sequences.append(n_gram_sequence)

    return input_sequences


def create_dataset(csv_file_path: Path, index_to_word, word_to_index):
    """
    Generate the model dataset
    :param csv_file_path: A path to the csv file containing the artist name, song name and lyrics
    :return: Datasets
    """
    df = pd.read_csv(csv_file_path, header=None)
    keep_col = [0, 1, 2]

    artists_names = df.iloc[:, 0]
    songs_names = df.iloc[:, 1]
    lyrics = df.iloc[:, 2]
    # tokenized_ngrams = tokenize(lyrics, word_to_index)

    all_words = set([word for l in lyrics for word in l])
    all_pre_trained_words = set(word_to_index.keys())

    print(all_words.symmetric_difference(all_pre_trained_words))

    # max_sequence_len = max([len(x) for x in tokenized_ngrams])
    # tokenized_ngrams = np.array(pad_sequences(tokenized_ngrams, maxlen=max_sequence_len, padding='pre'))
    # print(tokenized_ngrams)
    #
    # return tokenized_ngrams


def parser(artist_name, song_name, lyrics):
    """
    Load the images and labels and convert it to tensors
    :param lyrics:
    :param song_name:
    :param artist_name:
    :return:
    """
    tf.print(artist_name)
    # print(f'{tf.print(artist_name)},{song_name}, {lyrics}')

    return artist_name, song_name, lyrics


class RNN(keras.Model):
    def __init__(self, input_dim: int, pretrained_weights, output_dim):
        super(RNN, self).__init__()
        self.embed1 = Embedding(input_dim=input_dim, output_dim=output_dim)
        self.embed1.set_weights(pretrained_weights)
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


def train(csv_p,
          train_file_name=None,
          test_file_name=None,
          BATCH_SIZE=32
          ):
    word2vec_model_name = 'glove-twitter-25'
    word2vec_model = gensim.downloader.load(word2vec_model_name)
    pretrained_weights = word2vec_model.vectors
    index_to_word = word2vec_model.index_to_key
    word_to_index = word2vec_model.key_to_index
    tokenized = create_dataset(csv_p, index_to_word, word_to_index)

    # vocab_size, embedding_size = pretrained_weights.shape
    # output_dim = int(word2vec_model_name.split('-')[-1])
    #
    # RNN_model = RNN(input_dim=vocab_size, pretrained_weights=pretrained_weights,
    #                 output_dim=output_dim)
    # RNN_model.compile(optimizer=tf.keras.optimizers.Adam(),
    #                   loss="categorial_crossentropy",
    #                   metrics=["accuracy"])


csv_p = Path(rf'{csv_filename}')
train(csv_p)
