import sys
sys.path.append("../")
import numpy as np
from common import utils

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = utils.preprocess(text)

contexts, target = utils.create_contexts_target(corpus)
vocab_size = len(word_to_id)
target = utils.convert_one_hot(target, vocab_size)
contexts = utils.convert_one_hot(contexts, vocab_size)

print(contexts)
print(target)