import re
from functools import lru_cache
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


@lru_cache(maxsize=1)
def _english_stop_words() -> frozenset[str]:
   try:
      return frozenset(stopwords.words("english"))
   except LookupError:
      nltk.download("stopwords", quiet=True)
      return frozenset(stopwords.words("english"))


def preprocess_messages(messages: Iterable[object]) -> List[str]:
   english_stop_words = _english_stop_words()
   stemmer = PorterStemmer()
   processed = []
   for message in messages:
      words = re.sub("[^a-zA-Z]", " ", str(message or "")).lower().split()
      words = [stemmer.stem(word) for word in words if word not in english_stop_words]
      processed.append(" ".join(words))
   return processed
