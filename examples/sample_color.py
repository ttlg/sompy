import numpy as np
from sompy import SOM
import matplotlib.pyplot as plt


N = 20

input_layer = np.random.rand(10000, 3)
output_shape = (N, N)

som = SOM(output_shape, input_layer)

output_layer = som.train(100000)

plt.imshow(output_layer,
           interpolation='none')
plt.show()
