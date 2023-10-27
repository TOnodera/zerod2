from nptyping import NDArray

class SGD:
    def __init__(self, lr = 0.01) -> None:
        self.lr = lr

    def update(self, params: NDArray, grads: NDArray) -> None:
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
