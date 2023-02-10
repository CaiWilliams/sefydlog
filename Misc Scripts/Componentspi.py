
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
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

            polluted_df['Energy_Loss'] = (((polluted_df['Pmax'] - pristine_df['Pmax'])))    #/(polluted_df[sheet_names[idx]])
            polluted.append(polluted_df)
        polluted = pd.concat(polluted)
        #polluted = polluted.fillna(0)
        mask = (polluted.index > start_date) & (polluted.index <= end_date)
        polluted = polluted[mask]
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    print(polluted)
    return polluted.index, data


def pi(loc,start_date,end_date):
    n = ['carbon_monoxide', 'sulfur_dioxide', 'nitric_acid', 'water_vapour', 'methane', 'TAU550', 'formaldehyde', 'air_temperature', 'nitrogen_dioxide', 'ozone', 'carbon_dioxide', 'surface_air_pressure','relative_humidity']
    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True)

    data = []
    for idx,i in enumerate(n):
        dates, temp_data = calc_multiyear([loc],i,idx,start_date,end_date)
        data.append(temp_data)
    data = np.asarray(data)
    print(data)
    data_sum = np.zeros(13,)
    #for i in range(np.shape(data)[2]):
    #    data_sum[:] = data_sum[:] + data[:,:,i].flatten()
    #data_sum = data_sum/np.shape(data)[2]
    #data_sum = np.nan_to_num(data_sum,0)
    data_sum = np.nansum(data,axis=2).ravel()
    data_sum = np.delete(data_sum,[3,7,10,11,12])
    n = np.delete(n,[3,7,10,11,12])
    #data = np.sum(data,axis=1)
    plt.bar(n,data_sum)
    #plt.yscale('symlog')


pi('Redding_California','1/01/2020','1/12/2020')
pi('Sacramento_California','1/01/2020','1/12/2020')
plt.show()