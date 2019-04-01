def padding(data, fill):
    max_len = 0

    for block in data:
        size = len(block)
        max_len = max(size, max_len)

    for i in range(len(data)):
        sent_len = len(data[i])

        if sent_len < max_len:
            data[i] += [fill for i in range(max_len - sent_len)]

    return max_len


def get_padded_chars(data):

    unique_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', ';', '.', '!', '.', '?', ':', '\'', '\"',
                    '/', '\\', '|', '_', '@', '#', '$', '%', '^', '&', '*', '~']

    char_dict = dict()

    max_word_len = 7

    padded_char_sentences = []

    for sentence in data:
        for word in sentence:
            if word != '<None>':
                size = len(word)
                max_word_len = max(size, max_word_len)
                for char in word:
                    if char not in unique_chars:
                        unique_chars.append(char)

    unique_chars.sort()
    for i, char in enumerate(unique_chars):
        char_dict[char] = i + 1

    for sentence in data:
        padded_char_sentence = []
        for word in sentence:
            if word == '<None>':
                padded_char_sentence.append([0 for i in range(max_word_len)])
            else:
                size = len(word)
                chars = [char_dict[c] for c in word]
                chars += [0 for i in range(max_word_len - size)]
                padded_char_sentence.append(chars)

        padded_char_sentences.append(padded_char_sentence)

    return padded_char_sentences, max_word_len, char_dict
