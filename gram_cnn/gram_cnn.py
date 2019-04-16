from time import time
from keras.layers import Embedding, Conv2D, Conv1D, TimeDistributed, Concatenate, Reshape, Lambda, Dropout
from keras.models import Input, Model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback,LearningRateScheduler
from keras import optimizers
import math
import numpy as np
import keras.backend as K
from data_management import file_utils


def get_model(max_len_sentences, max_len_words, wv_size, config, tags, char_vocab=20, cnn_emb_size=10):

    char_inputs = Input(shape=(max_len_sentences * max_len_words,), name="char_input")

    char_vec = Embedding(input_dim=char_vocab, output_dim=cnn_emb_size,)(char_inputs)

    layers = []

    char_filters = config['char_filters']
    word_filters = config['word_filters']
    char_kernels = config['char_kernels']
    word_kernels = config['word_kernels']

    embed_size = sum(config['char_filters']) + wv_size

    char_vec = Reshape((max_len_sentences * max_len_words, cnn_emb_size, 1))(char_vec)

    for i, f in enumerate(char_filters):
        cnn_sub = Conv2D(filters=f, kernel_size=[char_kernels[i], cnn_emb_size],
                                         padding='valid', activation='tanh')(char_vec)
        cnn_sub = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(cnn_sub)
        cnn_sub = Reshape([-1, f])(cnn_sub)
        cnn_sub = Dropout(rate=0.25)(cnn_sub)
        layers.append(cnn_sub)

    cnn_out = Concatenate(axis=2)(layers)

    word_vec = Input(shape=(max_len_sentences, wv_size), name="w2v_input")

    concatenated = Concatenate(axis=2)([word_vec, cnn_out])

    concatenated = Reshape((max_len_sentences, embed_size, 1))(concatenated)



    layers = []

    for i, f in enumerate(word_filters):
        cnn_sub = TimeDistributed(Conv1D(filters=f, kernel_size=[word_kernels[i]],
                                         padding='valid', activation='tanh'),
                                  input_shape=[max_len_sentences, embed_size, 1]
                                  )(concatenated)
        cnn_sub = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(cnn_sub)
        cnn_sub = Reshape([max_len_sentences, f])(cnn_sub)
        cnn_sub = Dropout(rate=0.25)(cnn_sub)
        layers.append(cnn_sub)

    n_gram_out = Concatenate(axis=2)(layers)

    pos_vec = Input(shape=(max_len_sentences, 1), name="pos_input")

    n_gram_out = Concatenate(axis=2)([pos_vec, n_gram_out])

    classes = len(tags)

    crf = CRF(classes, sparse_target=True)

    crf_out = crf(n_gram_out)

    model = Model(inputs=[char_inputs, word_vec, pos_vec], outputs=crf_out)

    adam = optimizers.Adam(lr=0.002)

    model.compile(adam, loss=crf_loss, metrics=[crf_viterbi_accuracy])

    model.summary()

    return model


def main():

    training_path = '../data/NCBI_train_processed.pkl.gz'

    training_vec_path = '../data/NCBI_train_wv.pkl.gz'

    test_path = '../data/NCBI_test_processed.pkl.gz'

    test_vec_path = '../data/NCBI_test_wv.pkl.gz'

    print('Retrieving Training Data')
    training_data = file_utils.get_zipped_pkl_data(training_path)

    print('Retrieving Vectorized Data')
    training_vectorized = file_utils.get_zipped_pkl_data(training_vec_path)

    print('Retrieving Test Data')
    test_data = file_utils.get_zipped_pkl_data(test_path)

    max_len_sentences = training_data['max_sent']

    max_len_words = training_data['max_word']

    char_dict = training_data['char_dict']

    enc_tags = training_data['enc_tags']

    chars = training_data['char']

    pos = training_data['pos']

    test_tags = test_data['tags']

    char_vocab = len(char_dict) + 1

    config = {'char_kernels': [1, 3, 5, 7, 10], 'char_filters': [200, 200, 150, 150, 100],
              'word_kernels': [1, 3, 5, 7, 10], 'word_filters': [50, 50, 50, 50, 50]}

    print('Building Model')
    model = get_model(max_len_sentences, max_len_words, 200, config, test_tags, char_vocab)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=0, write_graph=True,
                              write_grads=True, write_images=True)

    filepath = '../data/saved-model-{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                                 save_best_only=False, mode='max')

    print('Training Model')
    model.fit(x=[np.array(chars), np.array(training_vectorized), np.array(pos)], epochs=1000, batch_size=32, y=np.array(enc_tags),
              validation_split=0.2, callbacks=[tensorboard, checkpoint], shuffle=True)


if __name__ == "__main__":
    main()