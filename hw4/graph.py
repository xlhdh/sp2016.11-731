import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

inputname = sys.argv[1]

data = np.loadtxt(inputname).T
#data = data[:,-3000:]
plt.subplot(121)
plt.plot(data[0], ',b', alpha=0.8, label='Train')
plt.grid()
plt.plot(data[1], 'g,', label='Test')
plt.xlim(0, data.shape[1]+1)
plt.legend()
plt.subplot(122)
data = data[1,-1000:]
plt.plot(data, 'g.', label='Final 1000')
plt.grid()
plt.savefig(inputname[:-4]+'.pdf', transparent=True)
plt.savefig(inputname[:-4]+'.png', transparent=True)

