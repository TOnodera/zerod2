import sys
sys.path.append("..")
from common import utils
import numpy as np

text = 'You say goodby and I say hello.'
corpus, word_to_id, id_to_word = utils.preprocess(text)
vocab_size = len(word_to_id)

C = utils.create_co_matrix(corpus, vocab_size)
W = utils.ppmi(C)

np.set_printoptions(precision=3)

print(C)
print("-"*100)
print(W)
