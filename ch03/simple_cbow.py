import sys
sys.path.append("../")
import numpy as np
from common import layers

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size) -> None:
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        
        # レイヤーの生成
        self.in_layer0 = layers.MatMul(W_in)
        self.in_layer1 = layers.MatMul(W_in)
        self.out_layer = layers.MatMul(W_out)
        self.loss_layer = layers.SoftmaxWithLoss()
        
        # すべての重みと勾配をリストにまとめる
        _layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in _layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout=1) -> None:
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)

        