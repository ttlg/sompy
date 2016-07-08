import numpy as np
from sompy import SOM
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 20
colors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
som = SOM((N, N), colors)
som.set_parameter(neighbor=0.3)
ims = []
for i in range(1000):
    m = som.train(10)
    img = np.array(m.tolist(), dtype=np.uint8)
    im = plt.imshow(m.tolist(), interpolation='none', animated=True)
    ims.append([im])
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
plt.show()
# ani.save('dynamic_images.mp4')
