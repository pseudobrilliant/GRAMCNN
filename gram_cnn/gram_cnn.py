from time import time
from keras.layers import Embedding, Conv2D, Conv1D, TimeDistributed, Highway, Concatenate, Reshape, Lambda, Dropout, Masking
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

    # Batch size x max_len_sentence x max_words
    char_inputs = Input(shape=(max_len_sentences, max_len_words,), name="char_input")

    # Batch size x max_len_sentence x max_words x char_embedding
    char_vec = Embedding(input_dim=char_vocab, output_dim=cnn_emb_size, mask_zero=0)(char_inputs)

    layers = []

    char_filters = config['char_filters']
    word_filters = config['word_filters']
    char_kernels = config['char_kernels']
    word_kernels = config['word_kernels']

    embed_size = sum(config['char_filters']) + wv_size

    # Batch size x max_len_sentence x max_words x char_embedding
    for i, f in enumerate(char_filters):
        # Batch size x max_len_sentence x 1 x Features
        cnn_sub = Conv2D(filters=f, kernel_size=[char_kernels[i], cnn_emb_size],
                                         padding='same', activation='tanh')(char_vec)
        # Batch size x max_len_sentence x 1 x Pool Features
        cnn_sub = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(cnn_sub)

        # Batch size x max_len_sentence x Pool Features
        cnn_sub = Reshape([max_len_sentences, f])(cnn_sub)
        cnn_sub = Dropout(rate=0.35)(cnn_sub)
        layers.append(cnn_sub)

    # Batch size x max_len_sentence x all features
    cnn_out = Concatenate(axis=2)(layers)

    # Batch size x max_sentence_x wv_embedding
    word_vec = Input(shape=(max_len_sentences, wv_size), name="w2v_input")

    # Batch size x max_len_sentence x total_embed
    concatenated = Concatenate(axis=2)([word_vec, cnn_out])

    layers = []

    # Batch size x max_len_sentence x total embed x 1
    concatenated = Reshape((max_len_sentences, embed_size, 1))(concatenated)

    for i, f in enumerate(word_filters):
        cnn_sub = Conv2D(filters=f, kernel_size=[word_kernels[i],embed_size],
                                         padding='same', activation='tanh')(concatenated)
        cnn_sub = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(cnn_sub)
        cnn_sub = Reshape([max_len_sentences, f])(cnn_sub)
        cnn_sub = Dropout(rate=0.35)(cnn_sub)
        layers.append(cnn_sub)

    n_gram_out = Concatenate(axis=2)(layers)

    n_gram_out = Masking(mask_value=0.0, input_shape=(max_len_sentences, embed_size))(n_gram_out)

    # Batch size x max_sentence x total_embed
    pos_vec = Input(shape=(max_len_sentences, 1), name="pos_input")

    # Batch size x max_sentence x total_embed + 1
    n_gram_out = Concatenate(axis=2)([pos_vec, n_gram_out])

    classes = len(tags)

    crf = CRF(classes, sparse_target=False)

    crf_out = crf(n_gram_out)

    model = Model(inputs=[char_inputs, word_vec, pos_vec], outputs=crf_out)

    opt = optimizers.Adam()

    model.compile(opt, loss=crf_loss, metrics=[crf_viterbi_accuracy])

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

    config = {'char_kernels': [1, 3, 5], 'char_filters': [200, 200, 200],
              'word_kernels': [1, 3, 5], 'word_filters': [50, 50, 50]}

    print('Building Model')
    model = get_model(max_len_sentences, max_len_words, 200, config, test_tags, char_vocab)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=0, write_graph=True,
                              write_grads=True, write_images=True)

    filepath = '../data/saved-model-{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                                 save_best_only=False, mode='max')

    print('Training Model')
    model.fit(x=[np.array(chars), np.array(training_vectorized), np.array(pos)], epochs=1000, batch_size=16, y=np.array(enc_tags),
              validation_split=0.2, callbacks=[tensorboard, checkpoint], shuffle=True)


if __name__ == "__main__":
    main()