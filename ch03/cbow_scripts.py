
import sys
sys.path.append("../")
import numpy as np
from common import layers

c0 = np.array([1, 0, 0, 0, 0, 0, 0,])
c1 = np.array([0, 0, 1, 0, 0, 0, 0,])

W_in = np.random.randn(7,3)
W_out = np.random.randn(3,7)

in_layer0 = layers.MatMul(W_in)
in_layer1 = layers.MatMul(W_in)
out_layer = layers.MatMul(W_out)

h0 =  in_layer0.forward(c0)
h1 =  in_layer1.forward(c1)
h = 0.5 * (h0+h1)
print(h, W_out)
s = out_layer.forward(h)

