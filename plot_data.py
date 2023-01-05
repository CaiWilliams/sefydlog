import matplotlib.pyplot as plt
import pandas as pd
import os


def plot(name):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    data = pd.read_csv(file)
    data = data['Water Vapour']
    return data

#plot('P3HTPCBM_Zhengzhou_2019')
A = plot('P3HTPCBM_Mexico_City_2019_1000HPa')
#plot('P3HTPCBM_Lahore_2019')
#plot('P3HTPCBM_San_Juan_2019')
B = plot('P3HTPCBM_Durham_2019')
C = A-B
plt.plot(C)
plt.show()