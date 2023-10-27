import sys
sys.path.append("../")
import matplotlib.pyplot as plt
from dataset import spiral

x, t =  spiral.load_data()
print('x', x.shape)
print('t', t.shape)

