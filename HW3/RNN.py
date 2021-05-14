import numpy as np
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

word2vec_model_name = 'glove-wiki-gigaword-300'
word2vec_model = gensim.downloader.load(word2vec_model_name)
pretrained_weights = word2vec_model.vectors
index_to_word = word2vec_model.index_to_key
word_to_index = word2vec_model.key_to_index

nltk.download('stopwords')


def apply_tf_tokenization(x):
    return tf.keras.preprocessing.text.text_to_word_sequence(
        str(x), filters='''!"#$%()*+,-./:;<=>?@[\\]^_`'{|}~\t\n''',
        lower=True, split=' ')


def add_words_from_songs(word2vec_model, songs_lyrics, word_to_index, output_dim):
    """
    Add words from all songs to the vocabulary
    :param output_dim: embedding output dimension
    :param word2vec_model: the word2vec model to update
    :param songs_lyrics: All songs lyrics
    :param word_to_index: Word to index dict
    :return: Updated word to index dict
    """

    all_words_from_songs = set()

    for song in songs_lyrics:
        curr = set(apply_tf_tokenization(song))
        all_words_from_songs = all_words_from_songs.union(curr)

    all_words_to_add = list(all_words_from_songs.difference(set(word_to_index.keys())))
    words_num = len(all_words_to_add)
    initial_weights = np.random.randn(*(words_num, output_dim))
    word2vec_model.add_vectors(keys=all_words_to_add, weights=initial_weights)


def tokenize(word_to_index, max_sequence_len):
    """ Tokenize the songs' lyrics
    :param songs_lyrics: All songs
    :param word_to_index: Word to index dict
    :return: Sequences of n-grams
    """

    current_words = set(word_to_index.keys())
    swords = set(stopwords.words("english"))
    w_set = current_words.difference(swords)
    num_of_classes = len(word_to_index)

    def tokenize_current_song(song):
        x = apply_tf_tokenization(song)
        x = [word_to_index[w] for w in x if w in w_set]
        offset_indices = x[1:] + [0]
        y = offset_indices

        return x, y

    return tokenize_current_song


def create_dataset(csv_file_path: Path, word2vec_model, index_to_word, word_to_index, output_dim):
    """
    Generate the model dataset
    :param output_dim:
    :param word2vec_model:
    :param word_to_index:
    :param index_to_word:
    :param csv_file_path: A path to the csv file containing the artist name, song name and lyrics
    :return: Predictors and labels
    """
    df = pd.read_csv(csv_file_path, header=None)
    artists_names, songs_names, songs_lyrics = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    max_sequence_len = max([len(s.split()) for s in songs_lyrics])
    add_words_from_songs(word2vec_model, songs_lyrics, word_to_index, output_dim)

    ds = tf.data.Dataset.from_tensor_slices(songs_lyrics)
    tokenize_func = tokenize(word_to_index, max_sequence_len)
    ds = ds.map(tokenize_func)
    ds_gen = ds.batch(3)

    for batch in ds_gen.as_numpy_iterator():
        print(ds_gen)

    # ds = tf.data.Dataset.from_tensor_slices((X, Y))
    # X = np.array(pad_sequences(X, maxlen=max_sequence_len, padding='pre'))

    return ds


class RNN(keras.Model):
    def __init__(self, input_dim: int, pretrained_weights, output_dim):
        super(RNN, self).__init__()
        self.embed1 = Embedding(input_dim=input_dim, output_dim=output_dim)
        self.embed1.build((None,))
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
    vocab_size, embedding_size = pretrained_weights.shape
    output_dim = int(word2vec_model_name.split('-')[-1])

    RNN_model = RNN(input_dim=vocab_size,
                    pretrained_weights=pretrained_weights,
                    output_dim=output_dim)
    ds = create_dataset(csv_path, word2vec_model, index_to_word, word_to_index, output_dim)
    print(ds)

    # RNN_model.compile(optimizer='adam',
    #                   loss="categorical_crossentropy",
    #                   metrics=["accuracy"])
    #
    # history = RNN_model.fit(predictors,
    #                         label,
    #                         epochs=epochs,
    #                         verbose=1,
    #                         batch_size=BATCH_SIZE)
    #
    # acc = history.history['accuracy']
    # loss = history.history['loss']
    # epochs = range(len(acc))
    # plt.plot(epochs, acc, 'b', label='Training accuracy')
    # plt.title('Training accuracy')
    # plt.figure()
    # plt.plot(epochs, loss, 'b', label='Training Loss')
    # plt.title('Training loss')
    # plt.legend()
    # plt.show()

    return ds


csv_filename = 'lyrics_train_set.csv'
csv_path = Path(rf'{csv_filename}')
X, Y = train(csv_path)