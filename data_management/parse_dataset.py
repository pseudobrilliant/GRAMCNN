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

    tokenized = []

    for sentence in sentences:
        words = nlp_utils.sentence_cleanup(sentence)
        tokenized_pos = nlp_utils.pos_tag(words)
        tokenized.append(tokenized_pos)

    return tokenized



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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--content", "-c", help="File to tokenize", required=True)
    parser.add_argument("--save", "-s", help="Save tokenized as", required=True)
    args = parser.parse_args()

    lines = get_lines(args.content)

    if lines[0] == '\n':
        lines.pop(0)

    excerpts = []
    categories = dict()
    tokenized = []

    while lines:

        sentences = get_sentences(lines)

        get_categories(lines, categories)

        tokenized.append(get_tokenized(sentences))

        excerpts.append(sentences)

    processed_data = {"excepts": excerpts, "categories": categories, "tokenized": tokenized}

    data_utils.zip_pkl_data(processed_data, args.save)

if __name__ == "__main__":
    main()
