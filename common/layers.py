from nptyping import NDArray
import numpy as np
from . import functions
from common import utils

class MatMul:
    def __init__(self, W: NDArray) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x: NDArray = None

    def forward(self, x: NDArray) -> NDArray:
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout: NDArray) -> NDArray:
        W, = self.params
        dx = np.dot(dout, W.T)
        dw = np.dot(self.x.T, dout)
        self.grads[0][...] = dw
        return dx



class Sigmoid:
    def __init__(self) -> None:
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x: NDArray) -> NDArray:
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: NDArray) -> NDArray:
        return dout * (1.0 - self.out) * self.out

class Affine:
    def __init__(self, W: NDArray, b:NDArray) -> None:
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x: NDArray = None

    def forward(self, x: NDArray) -> NDArray:
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout: NDArray) -> NDArray:
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)       
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = functions.softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = functions.cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class Embedding:
    def __init__(self, W: NDArray) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
        
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout) -> None:
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        
        
class EmbeddingDot:
    def __init__(self, W: NDArray) -> None:
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
        
    def forward(self, h: NDArray, idx: NDArray):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout: NDArray):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
    
class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoidの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = functions.cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx

class NegativeSamplingLoss:
    def __init__(self, W: NDArray, corpus, power=0.75, sample_size=5) -> None:
        self.sample_size = sample_size
        self.sampler = utils.UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
        
    def forward(self, h: NDArray, target: NDArray):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例のフォーワード
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # 負例のフォーワード
        negative_label = np.ones(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:,i]
            score = self.embed_dot_layers[1+i].forward(h, negative_target)
            loss += self.loss_layers[1+i].forward(score, negative_label)

        return loss
    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh
    
    