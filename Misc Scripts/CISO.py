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



def plot_generation():
    data_dir = os.path.join(os.path.dirname(os.getcwd()),'CISO-data')
    data_dirs = os.listdir(data_dir)
    data_dirs = [os.path.join(data_dir,dir) for dir in data_dirs if '.xml.zip' not in dir]
    data_dirs = [pd.read_csv(dir,compression='zip') for dir in data_dirs]
    data = pd.concat(data_dirs)

    data = data[data['RENEWABLE_TYPE'] == 'Solar']
    data = data.drop(columns=['LABEL','XML_DATA_ITEM','MARKET_RUN_ID_POS','RENEW_POS','MARKET_RUN_ID','GROUP','RENEWABLE_TYPE','INTERVALSTARTTIME_GMT','INTERVALENDTIME_GMT','OPR_INTERVAL'])

    data['OPR_HR'].values[data['OPR_HR'] >= 24] = 0
    Date = data['OPR_DT'] +' '+ data['OPR_HR'].astype(str)+':00:00'
    data = data.set_index(pd.DatetimeIndex(Date))
    data = data.drop(columns=['OPR_DT','OPR_HR'])

    d = data
    #d = data[data['TRADING_HUB'] == 'NP15' & data['TRADING_HUB'] == 'NP15']
    d = d.drop(columns=['TRADING_HUB'])
    d = d.sort_index()
    d = d.groupby(by=d.index).agg(sum)
    print(d)
    #d = d.iloc[12::24]
    #print(d)
    #d = d['MW'].to_numpy()
    #plt.plot(d)
    #plt.ylim(bottom=0)
    print(d.iloc[0].name)
    fires_start = dt.datetime(year=2020,month=8,day=16,hour=12)
    fires_end = dt.datetime(year=2020,month=9,day=25,hour=12)
    plt.xlim(d.iloc[0].name,d.iloc[-1].name)
    dates = [dt.datetime(year=2020,month=1,day=1,hour=0),dt.datetime(year=2020,month=4,day=1,hour=0),dt.datetime(year=2020,month=7,day=1,hour=0),dt.datetime(year=2020,month=10,day=1,hour=0),dt.datetime(year=2021,month=1,day=1,hour=0)]
    #plt.xticks(dates)
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.85, wspace=0.05, hspace=0.05)
    plt.ylabel('Photovoltaic Generation (MW)')
    return d

def plot_curtailment(gen):
    data_dir = os.path.join('Curtailment.csv')
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
    data['Solar Curtailment'] = data['Solar Curtailment']
    data = data.resample('24h').sum()
    temp = pd.concat([gen,data]).sort_index()
    temp = temp.groupby(temp.index).sum()

    temp['No Cut'] = temp['Solar Curtailment'] + temp['MW'].clip(lower=0)
    temp.to_csv('temp.csv')
    #temp.index = temp.index.duplicated(keep='first')
    x = data.index.to_numpy()
    y = data['Solar Curtailment'].to_numpy()
    #y1 = temp['MW'].clip(lower=0).to_numpy()
    plt.plot(x,y)



gen = plot_generation()
plot_curtailment(gen)

#plt.savefig('California_2020_CISO.png',dpi=1200)
#plt.savefig('California_2020_CISO.svg')
plt.show()
