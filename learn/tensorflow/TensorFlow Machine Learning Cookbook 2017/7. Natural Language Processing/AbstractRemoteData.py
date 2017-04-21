import string
from abc import ABCMeta, abstractmethod
import nltk
from nltk.corpus import stopwords


class AbstractRemoteData(ABCMeta):
    _stopwords_ = stopwords.words('english')

    def __new__(self, url: str = None):
        self._URL_ = url or 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

    def load_data(self):
        if not self.is_downloaded():
            contents = self.download()
            self.write_file(contents)

        df = self.read_data()
        return df

    @abstractmethod
    def is_downloaded(self): pass

    @abstractmethod
    def download(self): pass

    @abstractmethod
    def write_file(self, contents): pass

    @abstractmethod
    def read_data(self): pass

    @staticmethod
    def normalize_text(text: str) -> str:
        # Remove punctuation and digit
        char_filtered = ''.join(c for c in text if c not in string.punctuation and c not in string.digits)

        # Remove extra whitespaces and stopwords
        space_filtered = ' '.join(w for w in nltk.word_tokenize(char_filtered)
                                  if w not in AbstractRemoteData._stopwords_)
        return space_filtered.lower()
