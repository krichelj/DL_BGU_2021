from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pretty_midi
import datetime
from typing import List, Dict, Tuple

# nltk.download('stopwords')
word2vec_model_name = 'glove-wiki-gigaword-300'
word2vec_model = gensim.downloader.load(word2vec_model_name)
# word2vec_model = KeyedVectors.load_word2vec_format(path)
# word2vec_model = Word2Vec(vector_size=300).wv
print(f'Word2Vec model loaded')


def tf_text_to_tokens(text: str) -> List[str]:
    """
    Perform tokenization of the text by a set of specified rules using TensorFlow
    :param text: Text to tokenize
    :return: Tokens constructed from the text
    """
    tokens = text_to_word_sequence(input_text=text,
                                   filters='''!"#$%()*+,-./:;<=>?@[\\]^_`'{|}~\t\n''',
                                   lower=True,
                                   split=' ')
    return tokens


def add_words_from_songs(word_to_index, songs_lyrics: List[str], output_dim: int):
    """
    Add words from all songs to the vocabulary inplace
    :param word_to_index: The word2vec model word to index dictionary
    :param songs_lyrics: A list of all the songs lyrics
    :param output_dim: The embedding layer output dimension
    """
    all_words_from_songs = set()

    for song in songs_lyrics:
        curr = set(tf_text_to_tokens(song))
        all_words_from_songs = all_words_from_songs.union(curr)

    all_words_to_add = list(all_words_from_songs.difference(set(word_to_index.keys())))
    words_num = len(all_words_to_add)
    initial_weights = np.random.randn(*(words_num, output_dim))
    word2vec_model.add_vectors(keys=all_words_to_add, weights=initial_weights)


def recreate_w2v_model_with_pad_key(word2vec_model_keyed_vectors: KeyedVectors, embed_size: int) -> KeyedVectors:
    """
    Recreate the model with [PAD] word with the key 0
    :param word2vec_model_keyed_vectors: The word2vec model keyed vectors
    :param embed_size: The embedding layer dimension
    :return: The word2vec model keyed vectors with the pad key
    """
    existing_words = list(word2vec_model_keyed_vectors.key_to_index.keys())
    existing_embeddings = word2vec_model_keyed_vectors.vectors
    new_model_keyed_vectors = KeyedVectors(embed_size)
    new_model_keyed_vectors.add_vector(key='[PAD]', vector=np.zeros(embed_size))
    # word2vec_model.add_vector('[C-PAD]', np.zeros(embed_size))
    new_model_keyed_vectors.add_vectors(keys=existing_words, weights=existing_embeddings)
    return new_model_keyed_vectors


def tokenize(word_to_index: Dict[str, int], songs_lyrics: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
    """ Tokenize the songs lyrics to examples and labels
    :param word_to_index: The word2vec model word to index dictionary
    :param songs_lyrics: All songs lyrics
    :return: A list of examples and corresponding labels - each represented by a list of indices of words
    """

    current_words = set(word_to_index.keys())

    X = []
    Y = []

    for song in songs_lyrics:
        words_tokenized = tf_text_to_tokens(song)
        words = [word_to_index[w] for w in words_tokenized]
        x = words[:-1]  # example: we are going -> x = [we, are]
        y = words[1:]  # example: we are going -> y = [are, going]
        X.append(x)
        Y.append(y)

    return X, Y


def create_dataset(csv_file_path: Path, word_to_index: Dict[str, int], output_dim: int, batch_size: int = 32,
                   expand_vocab=True) \
        -> Dataset:
    """
    Generate the model dataset
    :param csv_file_path: A path to the csv file containing the artist name, song name and lyrics
    :param word_to_index: The word2vec model word to index dictionary
    :param output_dim: The embedding layer output dimension
    :param batch_size: The batch size
    :return: A TensorFlow Dataset object containing the examples and labels
    """
    df = pd.read_csv(csv_file_path, header=None)
    artists_names, songs_names, songs_lyrics = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    midi_embeddings_list, good_indices = generate_midi_embeddings(list(artists_names), list(songs_names))
    songs_lyrics = [songs_lyrics[i] for i in good_indices]
    if expand_vocab:
        add_words_from_songs(word_to_index, songs_lyrics, output_dim)

    X, Y = tokenize(word_to_index, songs_lyrics)

    def generator():
        for i, (row, labels) in enumerate(zip(X, Y)):
            yield {"input_1": row, "input_2": midi_embeddings_list[i]}, labels

    ds = Dataset.from_generator(generator=generator,
                                output_types=({"input_1": tf.int64, "input_2": tf.float32}, tf.int64))

    ds = ds.padded_batch(batch_size, padded_shapes=({"input_1": [None], "input_2": [128, None]}, [None]))

    # for x, y in ds.as_numpy_iterator():
    #     print(x, y)

    return ds


def generate_midi_embeddings(artists_names: List[str], songs_names: List[str]) -> Tuple[List[np.array], List[int]]:
    """
    Generate the MIDI files embeddings
    :param artists_names: A list of the artists names
    :param songs_names: A list of the songs names
    :return: A list of the piano roll and the corresponding list indices
    """
    piano_rolls = []
    good_indices = []

    for i, (artist_name, songs_name) in enumerate(zip(artists_names, songs_names)):
        # elton john, candle in the wind
        # Elton_John_-_Candle_in_the_Wind

        artist_name = '_'.join([x.capitalize() for x in artist_name.split()])
        songs_name = '_'.join([x.capitalize() for x in songs_name.split()])
        path = f'midi_files/{artist_name}_-_{songs_name}.mid'

        try:
            midi = pretty_midi.PrettyMIDI(path)
            # midi.remove_invalid_notes()
            piano_roll = midi.get_piano_roll(fs=1)  # the song is divided at each second
            piano_rolls.append(piano_roll)
            good_indices.append(i)

        except Exception as e:
            print(e)
            continue

    return piano_rolls, good_indices


class RNN(Model):
    """
    The base Recurrent Neural Network class
    Args:
      input_dim: The input dimension for the model - the vocabulary size
      pretrained_weights: The pretrained weights to be applied to the embedding layer
      output_dim: The output dimension for the model - the number of words in the word2vec model
      use_attention_addition: A boolean flag to indicate whether to use addition in attention or concatenation
    """

    def __init__(self, input_dim: int, pretrained_weights: np.array, output_dim: int,
                 use_attention_addition: bool = True):
        super(RNN, self).__init__()
        self.embed1 = Embedding(input_dim=input_dim, output_dim=output_dim, mask_zero=True)
        self.use_attention_addition = use_attention_addition
        self.embed1.build((None,))
        self.embed1.set_weights([pretrained_weights])
        self.embed1.trainable = False
        self.lstm1 = LSTM(units=256, return_sequences=True)
        # self.lstm2 = LSTM(units=256, return_sequences=True)
        self.dense = Dense(units=input_dim, activation='softmax')
        self.dense_for_midi = Dense(units=output_dim, activation='tanh')
        self.dense_for_concat = Dense(units=int(output_dim / 2), activation='relu')
        self.attention = Attention()
        self.bn = BatchNormalization(axis=1)  # normalize according to each pitch class (has 128) in the batch

    def call(self, inputs: Dict, training: bool = None, mask: bool = None):
        # input is a dict: {"input_1": [batch_size, input_length],
        #                   "input_2": [batch_size, 128, max_duration_of_song_in_batch]}

        # the embedding layer is pre-trained, thus we should not apply batch-norm

        song = inputs["input_1"]
        x = self.embed1(song)  # [batch_size, timestamps, embed_size]
        mask = self.embed1.compute_mask(song)

        piano_roll_mat = self.bn(inputs["input_2"], training=training)  # [batch_size, 128, max_duration_secs]
        # piano_roll_mat = inputs["input_2"]
        piano_roll_mat = tf.transpose(piano_roll_mat, perm=[0, 2, 1])  # [batch_size, max_duration_secs, 128]
        piano_roll_mat = self.dense_for_midi(piano_roll_mat)  # [batch_size, max_duration_secs, embed_size]

        # For each time step t in the song:
        #       calculate the attention with respect to each second in the song
        #       i.e: [second_vector*w_t for second_vector in piano_roll] where w_t is the word on time t
        # therefore we get attention batch of: [batch_size, timetsteps, embed_size]

        context = self.attention([x, piano_roll_mat], training=training)  # [batch_size, timetsteps, embed_size]
        if self.use_attention_addition:
            x = x + context
        else:
            x = tf.concat([x, context], axis=-1)  # [batch_size, timetsteps, 2*embed_size]
            # in order to reduce sparse dimensionality
            x = self.dense_for_concat(x)  # [batch_size, timetsteps, embed_size/2]

        x = self.lstm1(x, mask=mask)  # [batch_size, timesteps, units]
        # x = self.lstm2(x, mask=mask)  # [batch_size, timesteps, units]
        x = self.dense(x)  # [batch_size, timesteps, vocabulary_size]

        return x


def train(csv_path: Path, test_path: Path, batch_size: int = 32, epochs: int = 10):
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir = "logs/fit/" + time
    tensorboard_callback = TensorBoard(log_dir=logs_dir,
                                       histogram_freq=1)

    save_dir = f"model_save/rnn_{time}.hdf5"
    save_callback = ModelCheckpoint(filepath=save_dir,
                                    monitor="loss",
                                    verbose=1,
                                    save_best_only=True,
                                    mode="auto", period=1)

    output_dim = int(word2vec_model_name.split('-')[-1])

    global word2vec_model

    word2vec_model = recreate_w2v_model_with_pad_key(word2vec_model, output_dim)
    word_to_index = word2vec_model.key_to_index

    #for debug only!
    # ds = create_dataset(csv_path, word_to_index, output_dim, batch_size=2, expand_vocab=True)
    # _ = create_dataset(test_path, word_to_index, output_dim, batch_size=2, expand_vocab=True)
    ####

    # ds = create_dataset(csv_path, word_to_index, output_dim, batch_size)
    ds = create_dataset(csv_path, word_to_index, output_dim, batch_size=batch_size, expand_vocab=True)
    _ = create_dataset(test_path, word_to_index, output_dim, batch_size=batch_size, expand_vocab=True)

    pretrained_weights = word2vec_model.vectors
    vocab_size, embedding_size = pretrained_weights.shape

    RNN_model = RNN(input_dim=vocab_size,
                    pretrained_weights=pretrained_weights,
                    output_dim=output_dim)
    RNN_model.compile(optimizer='adam',
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

    # TODO: Trial - Erase
    RNN_model.run_eagerly = True

    # End trial

    history = RNN_model.fit(ds,
                            epochs=epochs,
                            steps_per_epoch=None,
                            verbose=1,
                            callbacks=[tensorboard_callback, save_callback])

    acc = history.history['accuracy']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.figure()

    loss = history.history['loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()


def generate_song(csv_path, test_csv_path, path_to_model):
    global word2vec_model

    output_dim = int(word2vec_model_name.split('-')[-1])

    # re-creating the model vocab
    word2vec_model = recreate_w2v_model_with_pad_key(word2vec_model, output_dim)
    word_to_index = word2vec_model.key_to_index
    # should update the model vocab
    ds = create_dataset(csv_path, word_to_index, output_dim, expand_vocab=True)
    X_test = create_dataset(test_csv_path, word_to_index, output_dim, batch_size=16, expand_vocab=True)


    pretrained_weights = word2vec_model.vectors
    vocab_size, embedding_size = pretrained_weights.shape

    # model = tf.keras.models.load_model(path_to_model)

    model = RNN(input_dim=vocab_size,
                pretrained_weights=pretrained_weights,
                output_dim=output_dim)

    model.compile(optimizer='adam',
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    x_build = None
    x_build_ds = ds.take(1)
    for x in x_build_ds:
        x_build = x[0]

    t = model(x_build)

    # loading the weights -
    model.load_weights(path_to_model)

    gen_songs = []

    index_to_word = word2vec_model.index_to_key

    word_to_index = word2vec_model.key_to_index

    def generate(song, song_midi):
        gen_song = song
        for i in range(50):
            inputs = {"input_1": gen_song, "input_2": song_midi["input_2"]}
            next_word = model.predict(inputs)  # [batch_size, i + 1, vocabulary_size]
            next_word = tf.expand_dims(next_word[:, -1, :], axis=1)  # taking the last prediction
            next_word = tf.math.argmax(next_word, axis=-1)  # [batch_size, 1]
            gen_song = tf.concat([gen_song, next_word], axis=-1)

        return gen_song

    w1_index = word_to_index['home']
    w2_index = word_to_index['all']
    w3_index = word_to_index['price']

    for song_midi, y in X_test:

        full_song = song_midi["input_1"]  # [batch_size, input_length]
        song = tf.expand_dims(full_song[:, 0], axis=-1)
        output_song = generate(song, song_midi)
        trials = [output_song]
        for w_ind in [w1_index, w2_index, w3_index]:
            s_tag = song.numpy()
            s_tag[:, -1] = w_ind
            # song = tf.expand_dims(song[:, 0], axis=-1)
            output_song = generate(s_tag, song_midi)
            trials.append(output_song)

        for gen_song in trials:
            for i, row in enumerate(gen_song.numpy()):
                lyrics = ' '.join([index_to_word[x] for x in row])
                print(f'({i}): {lyrics}')
                gen_songs.append(lyrics)

            print("=" * 80)

    return gen_songs


train_csv_filename = 'lyrics_train_set.csv'
train_path = Path(rf'{train_csv_filename}')
test_path = Path('lyrics_test_set.csv')
# train(train_path,  test_path)


path_to_model = 'model_save/rnn_20210522-140629.hdf5'

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
