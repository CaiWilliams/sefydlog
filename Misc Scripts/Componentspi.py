
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
import difflib
from scipy.interpolate import interp2d
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

def calc_multiyear(locations, n, idx, start_date, end_date):
    sheet_names = ['Carbon Monoxide', 'Sulfur Dioxide', 'Nitric Acid', 'Water Vapour', 'Methane', 'TAU550', 'Formaldehyde', 'Air Temperature', 'Nitrogen Dioxide', 'Ozone', 'Carbon Dioxide', 'Air Pressure','Relative Humididity']
    if n == 'surface_air_pressure':
        name = 'Air Pressure'
    else:
        name = difflib.get_close_matches(n, sheet_names, 3)[0]
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:

            pristine_dir = 'PERC_'+location+'_'+str(year)+'_relative_humidity'
            #pristine_dir = 'PERC_'+location+'_'+str(year)
            polluted_dir = 'PERC_'+location+'_'+str(year)+'_'+n

            pristine_df = split_date(fetch_df(pristine_dir)) 
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = (((polluted_df['Pmax']*10)) - (pristine_df['Pmax']*10)) / (pristine_df['Power'] * 1e9) * 100
            polluted.append(polluted_df)
        polluted = pd.concat(polluted)
        #polluted = polluted.fillna(0)
        mask = (polluted.index > start_date) & (polluted.index <= end_date)
        polluted = polluted[mask]
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data


def pi(loc,start_date):
    n = ['TAU550','nitrogen_dioxide','surface_air_pressure']
    #n = ['carbon_monoxide', 'sulfur_dioxide', 'nitric_acid', 'water_vapour', 'methane', 'TAU550', 'formaldehyde', 'air_temperature', 'nitrogen_dioxide', 'ozone', 'carbon_dioxide', 'surface_air_pressure','relative_humidity']
    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = start_date + dt.timedelta(hours=24)

    data = []
    for idx,i in enumerate(n):
        dates, temp_data = calc_multiyear([loc],i,idx,start_date,end_date)
        data.append(temp_data)
    data = np.asarray(data)
    data_sum = np.zeros(13,)
    #for i in range(np.shape(data)[2]):
    #    data_sum[:] = data_sum[:] + data[:,:,i].flatten()
    #data_sum = data_sum/np.shape(data)[2]
    #data_sum = np.nan_to_num(data_sum,0)
    data_sum = np.nanmean(data,axis=2).ravel()
    #data_sum = np.delete(data_sum,[3,7,10,12])
    #n = np.delete(n,[3,7,10,12])
    #data = np.sum(data,axis=1)
    return n,data_sum

def loop(loc,start_date,end_date):
    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True,)
    dates = pd.date_range(start_date,end_date,freq='24H')

    z = []
    for date in dates:
        sd = str(date.day) + '/' + str(date.month) + '/' + str(date.year)
        n,d = pi(loc,sd)
        z.append(d)
    z = np.asarray(z)
    plt.plot(dates,z)
    plt.xticks([dt.datetime(2020,8,16,12),dt.datetime(2020,9,5,12),dt.datetime(2020,9,25,12)])
    plt.ylabel('Daily Average Change in Energy Generation (%)')
    #plt.yticks(ticks=range(len(n)),labels=n,fontsize=9)
    #plt.colorbar()
    plt.tight_layout()

loop('Santa_Clarita_California','16/08/2020','25/09/2020')
plt.savefig('Santa_Clarita_California.png',dpi=600)
plt.clf()
loop('Fresno_California','16/08/2020','25/09/2020')
plt.savefig('Fresno_California.png',dpi=600)
plt.clf()
loop('Sacramento_California','16/08/2020','25/09/2020')
plt.savefig('Sacramento_California.png',dpi=600)
plt.clf()
loop('Redding_California','16/08/2020','25/09/2020')
plt.savefig('Redding_California.png',dpi=600)