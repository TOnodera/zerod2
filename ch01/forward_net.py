import numpy as np
from nptyping import NDArray

class Sigmoid:
    def __init__(self) -> None:
        self.params = []

    def forward(self, x: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-x))

class Affine:
    def __init__(self, W: NDArray, b: NDArray) -> None:
        self.params = [W, b]
    
    def forward(self, x: NDArray) -> NDArray:
        W, b = self.params
        return np.dot(x, W) + b

class TwoLayerNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # レイヤの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # すべての重みをリストにまとめる
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x: NDArray) -> NDArray:
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)


