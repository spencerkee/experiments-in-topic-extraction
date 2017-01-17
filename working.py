import operator
from collections import namedtuple

from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

import matplotlib.pyplot as plt

import pickle
import time
import unicodedata


# %matplotlib inline


# if __name__ == '__main__':
# 	main(name='test')