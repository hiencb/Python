import io
import os
import string
import requests
import tarfile
import pandas as pd


class MovieData:
    def __init__(self, url: str = None, tar_file: str = None, pos_file: str = None, neg_file: str = None):
        self._URL_ = url or 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
        self._TAR_FILE_ = tar_file or 'MovieReviews.tar.gz'
        self._POS_FILE_ = pos_file or 'PosMovieReviews.txt'
        self._NEG_FILE_ = neg_file or 'NegMovieReviews.txt'
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
        return (pos_content, neg_content)

    def write_file(self, contents):
        (pos_content, neg_content) = contents

        with open(self._POS_FILE_, 'w') as f:
            f.write(pos_content.read())

        with open(self._NEG_FILE_, 'w') as f:
            f.write(neg_content.read())

    def read_data(self):
        df = pd.read_table(self._FILE_, header=None, names=['Class', 'Content'])
        df['Content'] = df['Content'].map(self.normalize_text)
        df['Label'] = df['Class'].map(lambda c: 0 if c == 'ham' else 1)
        df['Text Length'] = df['Content'].map(lambda txt: len(txt.split()))
        return df

    @staticmethod
    def normalize_text(text: str) -> str:
        # Remove punctuation and digit
        char_filtered = ''.join(c for c in text if c not in string.punctuation and c not in string.digits)

        # Remove extra whitespaces
        space_filtered = ' '.join(w for w in char_filtered.split())
        return space_filtered.lower()
