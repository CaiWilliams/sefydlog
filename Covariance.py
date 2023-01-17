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

calc_cov('P3HTPCBM_Santa_Rosa_2020')