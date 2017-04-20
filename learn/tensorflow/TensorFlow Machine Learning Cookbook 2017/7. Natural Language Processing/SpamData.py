import os
import requests
import io
import string
import numpy as np
import pandas as pd
from zipfile import ZipFile


class SpamData:
    def __init__(self, url: str = None, file: str = None):
        self._URL_ = url or 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
        self._FILE_ = file or 'SpamCollection.txt'

    def download(self) -> str:
        res = requests.get(self._URL_)
        zipfile = ZipFile(io.BytesIO(res.content))
        file = zipfile.read('SMSSpamCollection')
        content = file.decode()
        return content

    def write_file(self, content: str):
        with open(self._FILE_, 'w') as f:
            f.write(content)

    def load_data(self):
        if not os.path.isfile(self._FILE_):
            content = self.download()
            self.write_file(content)

        df = self.create_dataframe()
        return df

    def create_dataframe(self) -> pd.DataFrame:
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
