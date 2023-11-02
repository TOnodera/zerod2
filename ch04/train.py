import sys
sys.path.append('../')
# from common import config
# config.GPU = True
import pickle
from common.np import GPU
from common.trainer import Trainer
from common.optimizers import Adam
from cbow import Cbow
from common import utils
from dataset import ptb
import numpy as np


# ハイパーパラメータの設定
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = utils.create_contexts_target(corpus, window_size)
if GPU:
    contexts, target = utils.to_gpu(contexts), utils.to_gpu(target)

# モデルなど生成
model = Cbow(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs

if GPU:
    word_vecs = utils.to_gpu(word_vecs)

params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)

