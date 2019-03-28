from keras.models import Input
from keras.layers import Embedding
from data_management import data_utils, file_utils

def get_model(max_len_sentences, max_len_words, wv_size, cnn_emb_size=25, char_vocab):

    word_inputs = Input(shape=(max_len_sentences, wv_size,), name= "w2v_input")

    char_inputs = Input(shape=(max_len_words,), name="char_input")

    cnn_emb = Embedding(input_dim=char_vocab, output_dim=cnn_emb_size,)(char_inputs)




def main():

    training_path = '../data/NCBI_train_processed.pkl.gz'

    training_data = file_utils.get_zipped_pkl_data(training_path)

    max_len_sentences = training_data['max_sent']

    max_len_words = training_data['max_word']

    char_dict = training_data['char_dict']

    wv_size = 200

    char_vocab = len(char_dict) + 1

    get_model(max_len_sentences, max_len_words, wv_size, char_vocab)

    return


if __name__ == "__main__":
    main()