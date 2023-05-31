import pandas as pd
import zipfile
import numpy as np
import gzip
import matplotlib.pyplot as plt
import matplotlib
import tkinter
matplotlib.use('TkAgg')
import os
import datetime as dt



def generation(dir):
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'CISO-data', 'Prod&Curt', dir)
    data = pd.read_csv(data_dir, delimiter=',')
    data = data.drop(columns=['Load','Wind','Net Load','Renewables','Nuclear','Large Hydro','Imports','Generation','Thermal','Load Less (Generation+Imports)'])
    data['Interval'] = data['Interval'] - 1
    data['Interval'] = data['Interval'] / 12 * 60
    data['Date'] = data['Date'].astype(str)
    data['Hour'] = data['Hour'].astype(int) - 1
    data['Interval'] = data['Interval'].astype(int)
    data['DateTime'] = data['Date'].astype(str) + ' ' + data['Hour'].astype(str) + ':' + data['Interval'].astype(str)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data = data.set_index('DateTime')
    data = data.drop(columns=['Date', 'Hour', 'Interval'])
    data['Solar'] = data['Solar'].fillna(0).astype(float)
    #data = data.sort_index()
    data = data.resample('24h').mean()
    return data
def curtailment(dir):
    data_dir = os.path.join(os.path.dirname(os.getcwd()),'CISO-data','Prod&Curt',dir)
    data = pd.read_csv(data_dir,delimiter=',')
    data = data.drop(columns=['Wind Curtailment'])
    data['Interval'] = data['Interval'] - 1
    data['Interval'] = data['Interval']/12 * 60
    data['Date'] = data['Date'].astype(str)
    data['Hour'] = data['Hour'].astype(int) - 1
    data['Interval'] = data['Interval'].astype(int)
    data['DateTime'] = data['Date'].astype(str) +' ' + data['Hour'].astype(str) + ':' + data['Interval'].astype(str)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data = data.set_index('DateTime')
    data = data.drop(columns=['Date','Hour','Interval'])
    data['Solar Curtailment'] = data['Solar Curtailment'].fillna(0).astype(float)
    #data = data.sort_index()
    data = data.resample('24h').mean()
    return data

x = generation('Generation_2020.csv')
y = curtailment('Curtailment_2020.csv')
z = x.assign(**y).fillna(0)
z.to_csv('temp3.csv')
plt.stackplot(z.index,z['Solar Curtailment'])
#plt.savefig('California_2020_CISO.png',dpi=1200)
#plt.savefig('California_2020_CISO.svg')
plt.show()
