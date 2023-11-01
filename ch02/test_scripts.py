import sys
sys.path.append("..")
from common import utils
import numpy as np
import matplotlib.pyplot as plt
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')

window_size = 2
wordvec_size = 100
vocab_size = len(word_to_id)
print('共起行列を生成')
C = utils.create_co_matrix(corpus, vocab_size, window_size)
print('共起行列をから相互情報量の行列に変換')
W = utils.ppmi(C, verbose=True)

print('SVD(特異値分解)による次元削減を実行')
try:
    from sklearn.utils import extmath
    U, S, V = extmath.randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    print("===Import Error===")
    U, S, V = np.linalg.svd(W)
    print('numpyのsvd()を実行しました')
    
word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    utils.most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


