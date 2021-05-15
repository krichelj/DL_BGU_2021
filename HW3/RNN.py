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

nltk.download('stopwords')
word2vec_model_name = 'glove-wiki-gigaword-300'
# word2vec_model = gensim.downloader.load(word2vec_model_name)
path = '/Users/nitzan/gensim-data/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz'
print(f'Loading word2vec model...')
# word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path)
word2vec_model = gensim.models.Word2Vec(vector_size=300)
word2vec_model = word2vec_model.wv
index_to_word = word2vec_model.index_to_key
word_to_index = word2vec_model.key_to_index


def apply_tf_tokenization(x):
    output = tf.keras.preprocessing.text.text_to_word_sequence(
        x, filters='''!"#$%()*+,-./:;<=>?@[\\]^_`'{|}~\t\n''',
        lower=True, split=' ')
    return output


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


def recreate_w2v_model_with_padkey(word2vec_model, embed_size):
    existing_words = list(word2vec_model.key_to_index.keys())
    existing_embeddings = word2vec_model.vectors
    new_model = gensim.models.KeyedVectors(embed_size)
    new_model.add_vector(key='[PAD]', vector=np.zeros(embed_size))
    # word2vec_model.add_vector('[C-PAD]', np.zeros(embed_size))
    new_model.add_vectors(keys=existing_words, weights=existing_embeddings)
    return new_model


def tokenize(songs_lyrics, word_to_index, max_sequence_len):
    """ Tokenize the songs' lyrics
    :param songs_lyrics: All songs
    :param word_to_index: Word to index dict
    :return: Sequences of n-grams
    """

    current_words = set(word_to_index.keys())
    swords = set(stopwords.words("english"))
    w_set = current_words.difference(swords)
    num_of_classes = len(word_to_index)

    X = []
    Y = []

    for song in songs_lyrics:
        x = apply_tf_tokenization(song)
        x = [word_to_index[w] for w in x if w in w_set]
        x = x[:-1]  # example: we are going -> x= [we, are]
        offset_indices = x[1:] # example: we are going -> x= [are, going]
        y = offset_indices
        X.append(x)
        Y.append(y)

    return X, Y


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

    # ## Trial
    # songs_lyrics = ['trial hello', 'hello trial']

    max_sequence_len = max([len(s.split()) for s in songs_lyrics])
    add_words_from_songs(word2vec_model, songs_lyrics, word_to_index, output_dim)
    X, Y = tokenize(songs_lyrics, word_to_index, max_sequence_len)

    X = np.array(pad_sequences(X, maxlen=max_sequence_len, padding='post', value=0))
    Y = np.array(pad_sequences(Y, maxlen=max_sequence_len, padding='post', value=0))

    return X, Y


class RNN(keras.Model):
    def __init__(self, input_dim: int, pretrained_weights, output_dim):
        super(RNN, self).__init__()
        self.embed1 = Embedding(input_dim=input_dim, output_dim=output_dim,
                                mask_zero=True)
        self.embed1.build((None,))
        self.embed1.set_weights([pretrained_weights])
        self.embed1.trainable = False
        self.lstm1 = LSTM(units=256, return_sequences=True)
        self.lstm2 = LSTM(units=256, return_sequences=True)
        self.dense = Dense(units=input_dim, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        # input: [batch_size, input_length]

        x = self.embed1(inputs)  # [batch_size, timesteps, units]
        mask = self.embed1.compute_mask(inputs)
        x = self.lstm1(x, mask=mask)  # [batch_size, timesteps, units]
        x = self.lstm2(x, mask=mask)  # [batch_size, timesteps, units]
        x = self.dense(x)  # [batch_size, timesteps, vocabulary_size]

        return x


def train(csv_path,
          BATCH_SIZE=32,
          epochs=10
          ):
    output_dim = int(word2vec_model_name.split('-')[-1])
    # output_dim = 300
    global word2vec_model

    word2vec_model = recreate_w2v_model_with_padkey(word2vec_model, output_dim)

    X, Y = create_dataset(csv_path, word2vec_model, index_to_word, word_to_index,
                          output_dim)

    pretrained_weights = word2vec_model.vectors
    vocab_size, embedding_size = pretrained_weights.shape

    RNN_model = RNN(input_dim=vocab_size,
                    pretrained_weights=pretrained_weights,
                    output_dim=output_dim)

    RNN_model.compile(optimizer='adam',
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

    ##TODO: Trial - Erase
    RNN_model.run_eagerly = True

    # x = tf.random.uniform(shape=(2, 3), minval=1, maxval=7210, dtype=tf.int32)
    # x = tf.constant([[2, 1, 0], [2, 2, 0]])

    # x = tf.constant([[2, 1, 1, 0], [2, 2, 1, 1]])
    # y_pred = RNN_model.predict(x)  # 2,3,2
    #
    # # hello world world , hello hello world world
    # y_true = [[1, 1, 0, 0], [2, 1, 1, 0]]
    #
    # y_pred_take_real = y_pred[:, [0, 1], :]
    # y_very_true = [[1, 1], [2, 1]]
    # sparse = tf.keras.losses.SparseCategoricalCrossentropy()
    #
    # sce1 = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    #
    # loss1 = sparse(y_true, y_pred)
    #
    # expected_loss = sparse(y_very_true, y_pred_take_real)
    #
    # eval_loss, acc = RNN_model.evaluate(x, tf.constant(y_true))

    ## End trial

    history = RNN_model.fit(X,
                            Y,
                            epochs=epochs,
                            verbose=1,
                            batch_size=BATCH_SIZE)

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

    return X, Y


csv_filename = 'lyrics_train_set.csv'
csv_path = Path(rf'{csv_filename}')
train(csv_path)
