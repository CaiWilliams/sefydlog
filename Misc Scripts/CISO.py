import pandas as pd
import zipfile
import numpy as np
import gzip
import matplotlib.pyplot as plt
import os
import datetime as dt

data_dir = os.path.join(os.getcwd(),'CISO-data')
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

d = data[data['TRADING_HUB'] == 'NP15']
d = d.drop(columns=['TRADING_HUB'])
d = d.sort_index()
d = d.iloc[12::24]
#print(d)
#d = d['MW'].to_numpy()
plt.plot(d,color='black')
print(d.iloc[0].name)
fires_start = dt.datetime(year=2020,month=8,day=16,hour=12)
fires_end = dt.datetime(year=2020,month=9,day=25,hour=12)
plt.axvspan(fires_start,fires_end,facecolor='tab:red',alpha=0.5)
plt.xlim(d.iloc[0].name,d.iloc[-1].name)
dates = [dt.datetime(year=2020,month=1,day=1,hour=0),dt.datetime(year=2020,month=4,day=1,hour=0),dt.datetime(year=2020,month=7,day=1,hour=0),dt.datetime(year=2020,month=10,day=1,hour=0),dt.datetime(year=2021,month=1,day=1,hour=0)]
plt.xticks(dates)
plt.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.85, wspace=0.05, hspace=0.05)
plt.ylabel('Photovoltaic Generation (MW)')
plt.savefig('California_2020_CISO.png',dpi=1200)
plt.savefig('California_2020_CISO.svg')
