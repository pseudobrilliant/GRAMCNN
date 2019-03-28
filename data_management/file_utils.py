import gzip
import logging
import requests
from gzip import GzipFile
from urllib.request import urlopen
import shutil
import os.path
import pickle
import urllib
import zipfile
from io import BytesIO


def fetch_url(url, dest):

    print("Requesting ... " + url)

    path_split = url.split('/')

    path = dest + path_split[-1]

    response = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    return path


def fetch_unpack_zip(url, dest):
    print("Requesting ... " + url)

    response = urllib.request.urlopen(url)
    compressed_file = BytesIO(response.read())
    zip_file = zipfile.ZipFile(compressed_file)
    zip_file.extractall(dest)

    print("Fetched ... " + url)


def get_zipped_pkl_data(gzip_path):
    """Retrieves data from uncompressed pickle file"""

    pkl_path = gzip_path.replace('.gz','')

    # If the pickle file is not available pull it from the compressed gzip file in data folder
    if not os.path.isfile(pkl_path):
        try:

            # Opens gzip file at data path and attempts to decompress into pickle file
            with open(gzip_path, 'rb') as zipf:
                compressed_file = BytesIO(zipf.read())
                decompressed_file = GzipFile(fileobj=compressed_file)

                with open(pkl_path, 'wb') as pkl:
                    shutil.copyfileobj(decompressed_file, pkl)

        except Exception as e:
            logging.error("Unable to unzip and read " + gzip_path + " \nWith error " + str(e))
            exit(1)

    file = open(pkl_path, 'rb')
    data = pickle.load(file)
    file.close()

    os.remove(pkl_path)

    return data


def zip_pkl_data(data, gzip_path):

    with gzip.GzipFile(gzip_path, 'wb') as zipf:

        zipf.write(pickle.dumps(data, 1))
