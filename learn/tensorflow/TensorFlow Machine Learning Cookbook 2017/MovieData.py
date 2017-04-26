import os
import tarfile
import requests
import nltk
import pandas as pd

from AbstractRemoteData import AbstractRemoteData


class MovieData(AbstractRemoteData):
    def __init__(self, url: str = None, tar_file: str = None, pos_file: str = None, neg_file: str = None, min_word_count: int = None):
        self._URL_ = url or 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
        self._TAR_FILE_ = tar_file or 'MovieReviews.tar.gz'
        self._POS_FILE_ = pos_file or 'PosMovieReviews.txt'
        self._NEG_FILE_ = neg_file or 'NegMovieReviews.txt'
        self._MIN_WORD_COUNT_ = min_word_count or 3
        self._CHUNK_SIZE_ = 2 ** 14

    def load_data(self):
        if not self.is_downloaded():
            contents = self.download()
            self.write_file(contents)

        df = self.read_data()
        return df

    def is_downloaded(self):
        return os.path.isfile(self._POS_FILE_) and os.path.isfile(self._NEG_FILE_)

    def download(self):
        res = requests.get(self._URL_, stream=True)

        with open(self._TAR_FILE_, 'wb') as f:
            for chunk in res.iter_content(chunk_size=self._CHUNK_SIZE_):
                if chunk:
                    f.write(chunk)
                    f.flush()

        tar = tarfile.open(name=self._TAR_FILE_, mode='r:gz')
        pos_content = tar.extractfile('rt-polaritydata/rt-polarity.pos')
        neg_content = tar.extractfile('rt-polaritydata/rt-polarity.neg')
        return pos_content, neg_content

    def write_file(self, contents):
        (pos_content, neg_content) = contents

        with open(self._POS_FILE_, 'w') as f:
            f.write(pos_content.read().decode('latin1'))

        with open(self._NEG_FILE_, 'w') as f:
            f.write(neg_content.read().decode('latin1'))

    def read_data(self):
        pos_data = pd.read_table(self._POS_FILE_, header=None, names=['Content'])
        pos_data = pos_data[pos_data['Content'].str.split().str.len() >= self._MIN_WORD_COUNT_]
        pos_data['Content'] = pos_data['Content'].map(self.normalize_text)

        neg_data = pd.read_table(self._NEG_FILE_, header=None, names=['Content'])
        neg_data = neg_data[neg_data['Content'].str.split().str.len() >= self._MIN_WORD_COUNT_]
        neg_data['Content'] = neg_data['Content'].map(self.normalize_text)

        return pos_data, neg_data

    @staticmethod
    def count_words(text):
        return len(nltk.word_tokenize(text))