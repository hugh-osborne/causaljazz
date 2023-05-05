from pycausaljazz import pycausaljazz
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

x = np.linspace(-2.0, 2.0, 100)
what = [a * (4.0/100) for a in norm.pdf(x, 0.0, 0.1)]

id = pycausaljazz.newDist([-2.0],[4.0],[100],what)
looped = pycausaljazz.readDist(id)
fig, ax = plt.subplots(1, 1)

ax.set_title('')
ax.plot(x, what)
ax.plot(x, looped)
fig.tight_layout()

plt.show()
