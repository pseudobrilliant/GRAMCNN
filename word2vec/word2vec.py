
from data_management import data_utils
from gensim.models import KeyedVectors


def get_word2vec(path):

    print("Loading Pre-Trained W2V")

    model = KeyedVectors.load_word2vec_format(path, binary=True)

    print("Loading Complete\n")

    return model


def get_features(model, data):

    print("Converting Features")

    features = []

    for excerpt in data:
        excerpt_feature = []
        for sentence in excerpt:
            for word in sentence:
                if word in model:
                    excerpt_feature.append(model[word])
        features.append(excerpt_feature)

    print("Features Converted\n")

    return features


def main():

    training_path = '../data/NCBI_train_processed.pkl.gz'

    test_path = '../data/NCBI_test_processed.pkl.gz'

    wv_path = '../data/wikipedia-pubmed-and-PMC-w2v.bin'

    training_data = data_utils.get_zipped_pkl_data(training_path)['words']

    model = get_word2vec(wv_path)

    features = get_features(model, training_data)

    data_utils.zip_pkl_data(features, '../data/NCBI_train_wv.pkl.gz')

    test_data = data_utils.get_zipped_pkl_data(test_path)['words']

    features = get_features(model, test_data)

    data_utils.zip_pkl_data(features, '../data/NCBI_test_wv.pkl.gz')

if __name__ == "__main__":
    main()