import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime

def plot(name):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    data = pd.read_csv(file)
    data = data['Pmax'].to_numpy()
    plt.plot(np.cumsum(data[::]))
    return data

def fetch(name,label):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    Data = pd.read_csv(file)
    data = Data[label]
    x = np.arange(len(data))
    xvals = np.arange(len(data)*3 -2)/3
    yvals = np.interp(xvals,x,data.to_numpy())
    xdates = pd.date_range(Data['Date'].loc[0],Data['Date'].loc[len(data)-1],freq='h')
    return xdates,yvals

def calc_power_diff(A, B, linestyle="-", colour='tab:blue'):
    fileA = os.path.join(os.path.dirname(os.getcwd()),'Results',A+'.csv')
    fileB = os.path.join(os.path.dirname(os.getcwd()),'Results',B+'.csv')
    dA = pd.read_csv(fileA)
    dB = pd.read_csv(fileB)
    data = dB['Pmax'] - dA['Pmax'])
    x = np.arange(len(data))
    xvals = np.arange(len(data)*3 -2)/3
    yvals = np.interp(xvals,x,data.to_numpy())
    xdates = pd.date_range(dA['Date'].loc[0],dA['Date'].loc[len(dA)-1],freq='h')
    return xdates,yvals



A = ''