import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import datetime as dt

def fetch_df(name):
    file = os.path.join(os.path.dirname(os.getcwd()),'Results',name+'.csv')
    data = pd.read_csv(file)
    return data

def split_date(df):
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df = df.set_index('Date')
    df.loc[:,'Hour'] = df.index.hour.values
    df.loc[:,'Day'] = df.index.day.values
    df.loc[:,'Month'] = df.index.month.values
    return df

def calc_multiyear(locations, start_date, end_date):
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:

            pristine_dir = 'PERC_'+location+'_pristine_'+str(year)
            polluted_dir = 'PERC_'+location+'_'+str(year)

            pristine_df = split_date(fetch_df(pristine_dir))
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = ((polluted_df['Pmax']*10) - (pristine_df['Pmax']*10))/(pristine_df['Power']*1e9) * 100
            #polluted_df['Energy_Loss'] = polluted_df['Energy_Loss'].clip(-100, 0).fillna(0)
            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    mask = (polluted.index >= start_date) & (polluted.index <=end_date)
    polluted = polluted[mask]
    return polluted


def compare_weeks(locs, period, central_start_date, periods):
    energy_lost = np.zeros((periods*2)+1)
    date = np.zeros((periods*2)+1,dtype=datetime.datetime)
    for idx,i in enumerate(range(-periods,periods+1)):
        start_date = central_start_date - dt.timedelta(days=period*i)
        end_date = start_date + dt.timedelta(days=period)
        data = calc_multiyear(locs, start_date,end_date)
        energy_lost[idx] = data['Energy_Loss'].mean()
        date[idx] = start_date
    plt.scatter(date,energy_lost)
    plt.show()
    return



#compare_weeks(['Beijing'],8,dt.datetime(day=8,month=8,year=2008,hour=0),32)

data = calc_multiyear(['Beijing_China'],dt.datetime(day=1,month=1,year=2008,hour=12),dt.datetime(day=31,month=12,year=2008,hour=12))
#for i in range(1,13):
t = data#[data.index.month == i]
t = t.groupby(t.index.dayofweek).mean()
plt.plot(['Monday','Tuesday','Wednesday','Thrusday','Friday','Saturday','Sunday'],t['Energy_Loss'])
plt.xlabel('Day of Week')
plt.ylabel('Average Electrical Power Loss (W/Wm$^{-2}$)')
plt.ylim(top=-8,bottom=-10)
#plt.legend()
#plt.plot(data['Energy_Loss'])
plt.savefig('Beijing_DaysOfWeek.png',dpi=600)
#plt.show()
#plt.twinx()
#plt.plot(data['PCE'].rolling(8*7,center=True).mean(),c='tab:orange')
#
# data = calc_multiyear(['Beijing'],dt.datetime(day=23,month=7,year=2008,hour=0),dt.datetime(day=8,month=8,year=2008,hour=0))
# plt.plot(np.cumsum(data['Energy_Loss'].to_numpy()))
#
# data = calc_multiyear(['Beijing'],dt.datetime(day=24,month=8,year=2008,hour=0),dt.datetime(day=9,month=9,year=2008,hour=0))
# plt.plot(np.cumsum(data['Energy_Loss'].to_numpy()))
