
from data_management import file_utils
from gensim.models import KeyedVectors
import numpy as np


def get_word2vec(path):

    print("Loading Pre-Trained W2V")

    model = KeyedVectors.load_word2vec_format(path, binary=True)

    print("Loading Complete\n")

    return model


def get_features(model, data):
    print(model.vector_size)
    print("Converting Features")

    features = []

    for sentence in data:
        sentence_feature = []
        for word in sentence:
            if word == '<None>' or word not in model:
                sentence_feature.append(np.array([0.0 for i in range(model.vector_size)]))
            else:
                sentence_feature.append(model[word])
        features.append(sentence_feature)

    print("Features Converted\n")

    return features


def main():

    training_path = '../data/NCBI_train_processed.pkl.gz'

    test_path = '../data/NCBI_test_processed.pkl.gz'

    wv_path = '../data/PubMed-and-PMC-w2v.bin'

    training_data = file_utils.get_zipped_pkl_data(training_path)['words']

    model = get_word2vec(wv_path)

    features = get_features(model, training_data)

    file_utils.zip_pkl_data(features, '../data/NCBI_train_wv.pkl.gz')

    del training_data
    del features

    test_data = file_utils.get_zipped_pkl_data(test_path)['words']

    features = get_features(model, test_data)

    file_utils.zip_pkl_data(features, '../data/NCBI_test_wv.pkl.gz')


if __name__ == "__main__":
    main()