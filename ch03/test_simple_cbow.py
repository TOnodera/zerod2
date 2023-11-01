import sys
sys.path.append("../")
from common.trainer import Trainer
from common.optimizers import Adam
from simple_cbow import SimpleCBOW
from common import utils

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbay and I say hello.'
corpus, word_to_id, id_to_word = utils.preprocess(text)
vocab_size = len(word_to_id)
contexts, target = utils.create_contexts_target(corpus, window_size)
contexts = utils.convert_one_hot(contexts, vocab_size)
target = utils.convert_one_hot(target, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)


trainer.fit(contexts, target, max_epoch, batch_size)

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

trainer.plot()
