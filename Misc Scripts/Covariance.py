import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from smarts import *

def calc_cov(A):
    fileA = os.path.join(os.getcwd(),'Results',A+'.csv')
    dA = pd.read_csv(fileA)
    atmos = dA.columns.to_numpy()[7:]
    atmos = np.insert(atmos,0,'PCE')
    atmos = np.delete(atmos,[2,8])
    pce = dA['PCE'].to_numpy()
    pce = np.argwhere(pce == 0).ravel()
    x = dA[atmos].to_numpy().T
    #x = np.delete(x, pce,axis=1)
    x = x[:,1800:2200]
    avg = np.average(x,axis=1)
    print(avg)
    zeros = np.argwhere(x == 0).ravel()
    x = np.delete(x,zeros,axis=1)
    x = np.corrcoef(x)
    plt.bar(atmos[1:],height=x[0][1:])
    #plt.yticks(np.arange(len(atmos)),labels=atmos)
    plt.show()
    return 

def calc_cov_multiyear(files):
    files = [os.path.join(os.getcwd(),'Results',str(f)+'.csv') for f in files]
    files_loaded = [pd.read_csv(f)for f in files]
    data = pd.concat(files_loaded)
    data.index = pd.DatetimeIndex(data['Date'])
    print(data)
    data = data.drop(columns=['Unnamed: 0','PCE','FF','Voc','Jsc','Air Pressure'])
    data = data[data.index.month == 7]
    data = data[data != 0]
    data = data.cov()
    data = data[data.index != 'Pmax']
    plt.plot(data['Pmax'])
    plt.show()

calc_cov_multiyear(['PERC_Santa_Rosa_2020'])