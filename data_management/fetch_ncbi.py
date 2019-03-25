import argparse
import urllib
import zipfile
from io import BytesIO
from urllib.request import urlopen
import os


def fetch_unpack(url):
    print("Requesting ... " + url)

    response = urllib.request.urlopen(url)
    compressed_file = BytesIO(response.read())
    zip_file = zipfile.ZipFile(compressed_file)
    zip_file.extractall('../data')

    print("Fetched ... " + url)


def main():

    parser = argparse.ArgumentParser()

    if not os.path.exists('../data'):
        os.makedirs('../data')

    content = 'https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip'

    parser.add_argument("--content", "-c", help="URL to pull content")
    args = parser.parse_args()

    if args.content:
        content = args.content

    fetch_unpack(content)


if __name__ == "__main__":
    main()
