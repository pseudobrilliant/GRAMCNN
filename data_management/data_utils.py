import gzip
import logging
from gzip import GzipFile
from io import BytesIO
import shutil
import os.path
import pandas as pd
import pickle

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

    data = pd.read_pickle(pkl_path)

    os.remove(pkl_path)

    return data


def zip_pkl_data(data, gzip_path):

    with gzip.GzipFile(gzip_path, 'wb') as zipf:

        zipf.write(pickle.dumps(data, 1))
