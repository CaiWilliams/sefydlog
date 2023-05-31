import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

longitudes = np.arange(-180,181,1)
longitude = (longitudes + 180) % 360 #- 180
plt.plot(longitudes,longitude)
plt.show()