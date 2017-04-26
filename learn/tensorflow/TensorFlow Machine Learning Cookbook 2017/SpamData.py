import io
import os
from zipfile import ZipFile
import pandas as pd
import requests

from AbstractRemoteData import AbstractRemoteData


class SpamData(AbstractRemoteData):
    def __init__(self, url: str = None, file: str = None):
        super().__init__()

        self._URL_ = url or 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
        self._FILE_ = file or 'SpamCollection.txt'

    def is_downloaded(self):
        return os.path.isfile(self._FILE_)

    def download(self) -> str:
        res = requests.get(self._URL_)
        zipfile = ZipFile(io.BytesIO(res.content))
        file = zipfile.read('SMSSpamCollection')
        content = file.decode()
        return content

    def write_file(self, content: str):
        with open(self._FILE_, 'w') as f:
            f.write(content)

    def read_data(self):
        df = pd.read_table(self._FILE_, header=None, names=['Class', 'Content'])
        df['Content'] = df['Content'].map(self.normalize_text)
        df['Label'] = df['Class'].map(lambda c: 0 if c == 'ham' else 1)
        df['Text Length'] = df['Content'].map(lambda txt: len(txt.split()))
        return df
