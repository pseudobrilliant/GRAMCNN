
from data_management import file_utils
from gensim.models import KeyedVectors
import numpy as np


def get_word2vec(path):

    print("Loading Pre-Trained W2V")

    model = KeyedVectors.load_word2vec_format(path, binary=True)

    print("Loading Complete\n")

    return model


def get_features(w2i, data):
    print("Converting Features")
    features = []

    for sentence in data:
        sentence_feature = []
        for word in sentence:
            if word == '<None>' or word not in w2i:
                sentence_feature.append(0)
            else:
                sentence_feature.append(w2i[word])
        features.append(sentence_feature)

    print("Features Converted\n")

    return features


def main():

    training_path = '../data/NCBI_train_processed.pkl.gz'

    test_path = '../data/NCBI_test_processed.pkl.gz'

    wv_path = '../data/wikipedia-pubmed-and-PMC-w2v.bin'

    training_data = file_utils.get_zipped_pkl_data(training_path)['words']

    print('Loading Model')

    model = get_word2vec(wv_path)

    print(model.vector_size)

    print(model.index2word[0])

    w2i = {w: w_index for w_index, w in enumerate(model.index2word)}

    features = get_features(w2i, training_data)

    file_utils.zip_pkl_data(features, '../data/NCBI_train_wv.pkl.gz')

    del training_data
    del features

    test_data = file_utils.get_zipped_pkl_data(test_path)['words']

    features = get_features(model, test_data)

    file_utils.zip_pkl_data(features, '../data/NCBI_test_wv.pkl.gz')

    file_utils.zip_pkl_data(w2i, '../data/NCBI_model_meta.pkl.gz')

if __name__ == "__main__":
    main()