import sys
sys.path.append("..")
from common.np import *
from dataset import sequence
from common.optimizers import Adam
from common.trainer import Trainer
from common.utils import eval_seq2seq
from attention_seq2seq import AttentionSeq2seq
# from ch07.seq2seq import Seq2seq
# from ch07.peeky_seq2seq import PeekySeq2seq

# データの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 入力文を反転
x_train, x_test = x_train[:,::-1], x_test[:,::-1]

# ハイパーパラメータの設定
vocab_size = len(char_to_id)
wordvec_size = 256
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

# モデル・オプティマイザー・トレイナー生成
model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 学習
acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose, is_reverse=True)
        
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

model.save_params()