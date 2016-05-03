import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

inputname = sys.argv[1]

data = np.loadtxt(inputname).T
plt.scatter(np.arange(data.shape[1]), data[0], 5, linewidth=0, alpha=0.8, label='Train')
plt.plot(data[1], 'g', label='Test')
plt.xlim(0, data.shape[1]+1)
plt.legend()
plt.grid()
plt.savefig(inputname[:-4]+'.png', transparent=False)
