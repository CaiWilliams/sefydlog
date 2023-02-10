import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot(name):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    data = pd.read_csv(file)
    data = data['PCE']
    plt.plot(data[::])
    return data

def plot_diff(A,B):
    fileA = os.path.join(os.getcwd(),'Results',A+'.csv')
    fileB = os.path.join(os.getcwd(),'Results',B+'.csv')
    dA = pd.read_csv(fileA)
    dB = pd.read_csv(fileB)
    data = ((dB['Pmax'] - dA['Pmax']) / dA['Pmax'])
    plt.plot(dB['Date'][::24], data[::24])
    plt.xticks(np.linspace(0,len(dB['Date'][::24])-4,3))
    plt.ylabel('Change in PCE due to Polution (%)')
    return data

# plot_diff('P3HTPCBM_Mexico_City_Pristine_2019','P3HTPCBM_Mexico_City_2019')
# plot_diff('P3HTPCBM_Zhengzhou_Pristine_2019','P3HTPCBM_Zhengzhou_2019')
# plot_diff('P3HTPCBM_Lahore_Pristine_2019','P3HTPCBM_Lahore_2019')
# plot_diff('P3HTPCBM_San_Juan_Pristine_2019','P3HTPCBM_San_Juan_2019')
# plot_diff('P3HTPCBM_Durham_Pristine_2019','P3HTPCBM_Durham_2019')

plot('P3HTPCBM_Mexico_City_2019')
# plot('P3HTPCBM_Zhengzhou_2019')
# plot('P3HTPCBM_Lahore_2019')
# plot('P3HTPCBM_Lahore_Pristine_2019')
# plot('P3HTPCBM_San_Juan_2019')
#plot('P3HTPCBM_Durham_Test_2019')

plt.show()