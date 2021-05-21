from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
import matplotlib.pyplot as plt
from tensorflow import keras
from nltk.corpus import stopwords
import nltk
import gensim.downloader
import numpy as np
import pretty_midi
import datetime

nltk.download('stopwords')
word2vec_model_name = 'glove-wiki-gigaword-300'
# word2vec_model = gensim.downloader.load(word2vec_model_name)
path = '/Users/nitzan/gensim-data/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz'
print(f'Loading word2vec model...')
# word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path)
word2vec_model = gensim.models.Word2Vec(vector_size=300)
word2vec_model = word2vec_model.wv


def get_index_to_word():
    # returns the most updated dict
    global word2vec_model
    return word2vec_model.index_to_key


def get_word_to_index():
    # returns the most updated dict
    global word2vec_model
    return word2vec_model.key_to_index


def apply_tf_tokenization(x):
    output = tf.keras.preprocessing.text.text_to_word_sequence(
        x, filters='''!"#$%()*+,-./:;<=>?@[\\]^_`'{|}~\t\n''',
        lower=True, split=' ')
    return output


def add_words_from_songs(word2vec_model, songs_lyrics, output_dim):
    """
    Add words from all songs to the vocabulary
    :param output_dim: embedding output dimension
    :param word2vec_model: the word2vec model to update
    :param songs_lyrics: All songs lyrics
    :param word_to_index: Word to index dict
    :return: Updated word to index dict
    """

    word_to_index = get_word_to_index()

    all_words_from_songs = set()

    for song in songs_lyrics:
        curr = set(apply_tf_tokenization(song))
        all_words_from_songs = all_words_from_songs.union(curr)

    all_words_to_add = list(all_words_from_songs.difference(set(word_to_index.keys())))
    words_num = len(all_words_to_add)
    initial_weights = np.random.randn(*(words_num, output_dim))
    word2vec_model.add_vectors(keys=all_words_to_add, weights=initial_weights)


def recreate_w2v_model_with_padkey(word2vec_model, embed_size):
    """
    Recreate the model with [PAD] word with 0 key.
    :param word2vec_model:
    :param embed_size:
    :return:
    """
    existing_words = list(word2vec_model.key_to_index.keys())
    existing_embeddings = word2vec_model.vectors
    new_model = gensim.models.KeyedVectors(embed_size)
    new_model.add_vector(key='[PAD]', vector=np.zeros(embed_size))
    # word2vec_model.add_vector('[C-PAD]', np.zeros(embed_size))
    new_model.add_vectors(keys=existing_words, weights=existing_embeddings)
    return new_model


def tokenize(songs_lyrics):
    """ Tokenize the songs' lyrics
    :param songs_lyrics: All songs
    :param word_to_index: Word to index dict
    :return: Sequences of n-grams
    """

    word_to_index = get_word_to_index()

    current_words = set(word_to_index.keys())
    swords = set(stopwords.words("english"))
    w_set = current_words.difference(swords)
    num_of_classes = len(word_to_index)

    X = []
    Y = []

    for song in songs_lyrics:
        words = apply_tf_tokenization(song)
        words = [word_to_index[w] for w in words if w in w_set]
        x = words[:-1]  # example: we are going -> x = [we, are]
        y = words[1:]  # example: we are going -> y = [are, going]
        X.append(x)
        Y.append(y)

    return X, Y


def create_dataset(csv_file_path: Path, word2vec_model, output_dim, BATCH_SIZE=1):
    """
    Generate the model dataset
    :param output_dim:
    :param word2vec_model:
    :param csv_file_path: A path to the csv file containing the artist name, song name and lyrics
    :return: Predictors and labels
    """
    df = pd.read_csv(csv_file_path, header=None)
    artists_names, songs_names, songs_lyrics = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]

    # ## Trial
    # songs_lyrics = ['trial hello', 'hello trial']

    midi_embeddings_list, good_indices = generate_midi_embeddings(list(artists_names)[:5], list(songs_names)[:2])

    songs_lyrics = [songs_lyrics[i] for i in good_indices]

    add_words_from_songs(word2vec_model, songs_lyrics, output_dim)

    X, Y = tokenize(songs_lyrics)

    def generator():
        for i, (row, labels) in enumerate(zip(X, Y)):
            yield {"input_1": row, "input_2": midi_embeddings_list[i]}, labels

    ds = tf.data.Dataset.from_generator(generator,
                                        output_types=({"input_1": tf.int64, "input_2": tf.float32}, tf.int64))

    ds = ds.padded_batch(BATCH_SIZE, padded_shapes=({"input_1": [None], "input_2": [128, None]}, [None]))

    # for x, y in ds.as_numpy_iterator():
    #     print(x, y)

    return ds


def generate_midi_embeddings(artists, songs):
    ret = []
    good_indices = []
    for i, (artist, song) in enumerate(zip(artists, songs)):
        ## elton john, candle in the wind
        ## Elton_John_-_Candle_in_the_Wind
        a = '_'.join([x.capitalize() for x in artist.split()])
        b = '_'.join([x.capitalize() for x in song.split()])
        path = f'midi_files/{a}_-_{b}.mid'
        try:
            midi = pretty_midi.PrettyMIDI(path)
            # midi.remove_invalid_notes()
            piano_matrix = midi.get_piano_roll(fs=1)  # the song is divided at each second
            ret.append(piano_matrix)
            good_indices.append(i)

        except Exception as e:
            print(e)
            continue

    return ret, good_indices


class RNN(keras.Model):
    def __init__(self, input_dim: int, pretrained_weights, output_dim, add=True):
        super(RNN, self).__init__()
        self.embed1 = Embedding(input_dim=input_dim, output_dim=output_dim,
                                mask_zero=True)
        self.add = add
        self.embed1.build((None,))
        self.embed1.set_weights([pretrained_weights])
        self.embed1.trainable = False
        self.lstm1 = LSTM(units=256, return_sequences=True)
        # self.lstm2 = LSTM(units=256, return_sequences=True)
        self.dense = Dense(units=input_dim, activation='softmax')
        self.dense_for_midi = Dense(units=output_dim, activation='tanh')
        self.dense_for_concat = Dense(units=int(output_dim / 2), activation='relu')
        self.attention = Attention()
        self.bn = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        # input is a dict: {"input_1": [batch_size, input_length], "input_2": [batch_size, 128, max_duration_of_song_in_batch]}

        # the embedding is pre-trained, thus we should not apply batch-norm
        song = inputs["input_1"]
        x = self.embed1(song)  # [batch_size, timesteps, embed_size]
        mask = self.embed1.compute_mask(song)

        piano_roll_mat = self.bn(inputs["input_2"])  # [batch_size, 128, max_duration_secs]
        piano_roll_mat = tf.transpose(piano_roll_mat, perm=[0, 2, 1])  # [batch_size, max_duration_secs, 128]
        piano_roll_mat = self.dense_for_midi(piano_roll_mat)  # [batch_size, max_duration_secs, embed_size]

        # For each time step t in the song:
        #       calculate the attention with respect to each second in the song
        #       i.e: [second_vector*w_t for second_vector in piano_roll] where w_t is the word on time t
        # therefore we get attention batch of: [batch_size, timetsteps, embed_size]
        context = self.attention([x, piano_roll_mat])  # [batch_size, timetsteps, embed_size]
        if self.add:
            x = x + context
        else:
            x = tf.concat([x, context], axis=-1)  # [batch_size, timetsteps, 2*embed_size]
            # in order to reduce sparse dimensionality
            x = self.dense_for_concat(x)  # [batch_size, timetsteps, embed_size/2]

        x = self.lstm1(x, mask=mask)  # [batch_size, timesteps, units]
        # x = self.lstm2(x, mask=mask)  # [batch_size, timesteps, units]
        x = self.dense(x)  # [batch_size, timesteps, vocabulary_size]

        return x


def train(csv_path,
          BATCH_SIZE=32,
          epochs=10
          ):
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir = "logs/fit/" + time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)

    save_dir = f"model_save/rnn_{time}.hp5"
    save_callback = tf.keras.callbacks.ModelCheckpoint(save_dir, monitor="loss", verbose=1, save_best_only=True,
                                                       mode="auto", period=1)

    output_dim = int(word2vec_model_name.split('-')[-1])

    global word2vec_model

    word2vec_model = recreate_w2v_model_with_padkey(word2vec_model, output_dim)

    ds = create_dataset(csv_path, word2vec_model, output_dim, BATCH_SIZE)

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

    ## End trial

    history = RNN_model.fit(ds,
                            epochs=epochs,
                            steps_per_epoch=None,
                            verbose=1,
                            callbacks=[tensorboard_callback, save_callback])

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

    # return X, Y


def generate_song(csv_path, test_csv_path, path_to_model):
    global word2vec_model

    output_dim = int(word2vec_model_name.split('-')[-1])

    # re-creating the model vocab
    word2vec_model = recreate_w2v_model_with_padkey(word2vec_model, output_dim)

    # should update the model vocab
    ds = create_dataset(csv_path, word2vec_model, output_dim)

    pretrained_weights = word2vec_model.vectors
    vocab_size, embedding_size = pretrained_weights.shape

    model = tf.keras.models.load_model(path_to_model)

    #Note: hdf5 is not saving how the layers are connected, thus we need to pass an input
    # model.build({"input_1": [None, None], "input_2": [None, 128, None]})

    # model.built = True

    # loading the weights -
    # model.load_weights(path_to_model)

    # should not change the word2vec model
    X_test, _ = create_dataset(test_csv_path, word2vec_model, output_dim)

    gen_songs = []

    index_to_word = get_index_to_word()

    for song in X_test:
        gen_song = [song[0]]
        for i in range(70):
            next_word = model.predict(tf.constant(gen_song))
            gen_song.append(next_word)
        lyrics = ' '.join([index_to_word[x] for x in gen_song])
        gen_songs.append(lyrics)

    return gen_songs


train_csv_filename = 'lyrics_train_set.csv'
train_path = Path(rf'{train_csv_filename}')
test_path = Path('lyrics_train_set.csv')
# train(train_path)

path_to_model = 'model_save/rnn_20210520-205817.hp5'

generate_song(train_path, test_path, path_to_model)


def play_with_midi():
    import pretty_midi
    path = "midi_files/adele_-_Hello.mid"
    midi = None
    try:
        midi = pretty_midi.PrettyMIDI(path)
        midi.remove_invalid_notes()
        print('')
    except Exception as e:
        print(e)
