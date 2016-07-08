import numpy as np
from sompy import SOM
import matplotlib.pyplot as plt


N = 40

input_layer = np.random.rand(10000, 3)
output_shape = (N, N)

som = SOM(output_shape, input_layer)
som.set_parameter(neighbor=0.1, learning_rate=0.2)

output_layer = som.train(10000)

plt.imshow(output_layer,
           interpolation='none')
plt.show()
