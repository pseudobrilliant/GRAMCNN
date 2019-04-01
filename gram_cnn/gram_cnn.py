from keras.layers import Embedding, Conv2D, MaxPool2D, Flatten, concatenate, Reshape
from keras.models import Input, Model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import numpy as np
from keras import backend as tf
from sklearn.preprocessing import OneHotEncoder
from data_management import file_utils


def get_model(max_len_sentences, max_len_words, wv_size, config, tags, char_vocab=26, cnn_emb_size=15):

    char_inputs = Input(shape=(max_len_sentences, max_len_words), name="char_input")

    char_vec = Embedding(input_dim=char_vocab, output_dim=cnn_emb_size,)(char_inputs)

    layers = []

    char_filters = config['char_filters']
    word_filters = config['word_filters']
    char_kernels = config['char_kernels']
    word_kernels = config['word_kernels']

    embed_size = sum(config['char_filters']) + wv_size

    for i, f in enumerate(char_filters):

        cnn_sub = Conv2D(filters= f , kernel_size= char_kernels[i], activation='tanh')(char_vec)
        cnn_sub = MaxPool2D(padding='same')(cnn_sub)
        cnn_sub = Reshape([-1, f])(cnn_sub)

        layers.append(cnn_sub)

    cnn_out = concatenate(layers, axis=1)

    word_vec = Input(shape=(max_len_sentences, wv_size,), name="w2v_input")

    concatenated = concatenate([word_vec, cnn_out], axis=1)

    concatenated = Reshape([-1, embed_size])(concatenated)

    concatenated = tf.expand_dims(concatenated, 0)

    layers = []

    for i, f in enumerate(word_filters):

        cnn_sub = Conv2D(filters=f, kernel_size=word_kernels[i], activation='tanh')(concatenated)
        cnn_sub = Reshape([-1, f])(cnn_sub)

        layers.append(cnn_sub)

    outputs = []

    for i in range(max_len_sentences):

        indices = tf.gather(layers[0], i)
        indices = tf.reshape(indices, [1, -1])
        n_grams = [indices]

        for j in word_kernels:
            if i == 0:
                n_grams.append(tf.reshape(tf.gather(layers[j - 1], i), [1, -1]))
            else:
                indices = list(range(i - j + 1 if i - j + 1 > 0 else 0,
                                     i + 1 if i + 1 < max_len_sentences - j + 1 else max_len_sentences - j + 1))
                gram_feature = tf.gather(layers[j - 1], indices)
                n_grams.append(tf.reshape(tf.max(gram_feature, axis=0, keepdims=True), [1, -1]))

        conc = tf.concatenate(n_grams, 1)

        outputs.append(conc)

    n_gram_out = tf.concatenate(outputs, 0)

    crf = CRF(len(tags), sparse_target=True)

    crf_out = crf(n_gram_out)

    model = Model(inputs=[char_inputs, word_inputs], outputs=[crf_out])

    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])

    model.summary()

    return model

def main():

    training_path = '../data/NCBI_train_processed.pkl.gz'

    vec_path = '../data/NCBI_train_wv.pkl.gz'

    training_data = file_utils.get_zipped_pkl_data(training_path)

    vectorized = file_utils.get_zipped_pkl_data(vec_path)

    max_len_sentences = training_data['max_sent']

    max_len_words = training_data['max_word']

    char_dict = training_data['char_dict']

    tags = training_data['tags']

    enc_tags = training_data['enc_tags']

    chars = training_data['char']

    wv_size = 200

    char_vocab = len(char_dict) + 1

    config = {'char_kernels': [1, 2, 3], 'char_filters': [200, 200, 200],
              'word_kernels': [1, 2, 3], 'word_filters': [50, 50, 50]}

    model = get_model(max_len_sentences, max_len_words,  wv_size, config, tags, char_vocab)

    model.fit([np.array(chars), np.array(vectorized)], np.array(enc_tags))


if __name__ == "__main__":
    main()