import argparse
import data_management.nlp_utils as nlp_utils
import data_management.data_utils as data_utils
import re


def clean(data):

    strip = re.sub('<category=.*?>', '', data)

    strip = strip.replace('</category>', '')

    return strip


def get_categories(content, categories):

    while True:

        if not content:
            return
        line = content[0]
        content.pop(0)

        if line == '\n':
            return
        else:
            values = line.split('\t')
            name = values[3]
            type = values[4]

            if name not in categories:
                categories[name] = [type]
            elif type not in categories[name]:
                categories[name].append(type)


def get_tokenized(sentences):

    post = []
    wordt = []
    for sentence in sentences:
        words = nlp_utils.sentence_cleanup(sentence)
        tokenized_pos = nlp_utils.pos_tag(words)
        post.append(tokenized_pos)
        wordt.append(words)

    return wordt, post


def get_sentences(content):

    sentences = []

    for i in range(2):
        line = content.pop(0)
        line = re.sub('\d+\|[a|t]\|', '', line)
        sentences += nlp_utils.sent_tokenize(line)

    return sentences


def get_lines(content_path):

    with open(content_path, 'r') as f:

        return f.readlines()


def parse_dataset(content, save):

    print("Parsing file at " + content)

    lines = get_lines(content)

    if lines[0] == '\n':
        lines.pop(0)

    excerpts = []
    categories = dict()
    tokenized_words = []
    tokenized_pos = []

    while lines:
        sentences = get_sentences(lines)

        get_categories(lines, categories)

        (twords, tpos) = get_tokenized(sentences)

        tokenized_words.append(twords)

        tokenized_pos.append(tpos)

        excerpts.append(sentences)

    processed_data = {"excepts": excerpts, "categories": categories, "words": tokenized_words, "pos": tokenized_pos}

    data_utils.zip_pkl_data(processed_data, save)

    print("Completed Parsing\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--content", "-c", help="File to tokenize", required=True)
    parser.add_argument("--save", "-s", help="Save tokenized as", required=True)
    args = parser.parse_args()

    parse_dataset(args.content, args.save)

if __name__ == "__main__":
    main()
