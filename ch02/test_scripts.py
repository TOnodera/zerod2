import sys
sys.path.append("..")
from common import utils
import numpy as np
import matplotlib.pyplot as plt

text = 'You say goodby and I say hello.'
corpus, word_to_id, id_to_word = utils.preprocess(text)
vocab_size = len(word_to_id)

C = utils.create_co_matrix(corpus, vocab_size)
W = utils.ppmi(C)

np.set_printoptions(precision=3)

U, S, V = np.linalg.svd(W)

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()