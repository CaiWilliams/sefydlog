
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
import datetime as dt
from math import cos, asin, sqrt

def fetch_df(name):
    file = os.path.join(os.path.dirname(os.getcwd()),'Results_errors',name+'.csv')
    data = pd.read_csv(file)
    return data

def split_date(df):
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df = df.set_index('Date')
    df.loc[:,'Hour'] = df.index.hour.values
    df.loc[:,'Day'] = df.index.day.values
    df.loc[:,'Month'] = df.index.month.values
    return df

def calc_distance(loc_lat,loc_lon,file):
    df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'Location Lists',file+'.csv'))
    name = df['Name'].to_numpy()
    lat = df['Latitude'].to_numpy()
    lon = df['Longitude'].to_numpy()
    dist = []
    for i in range(len(lat)):
        p = 0.017453292519943295
        hav = 0.5 - cos((lat[i] - loc_lat) * p) / 2 + cos(loc_lat * p) * cos(lat[i] * p) * (1 - cos((lon[i] - loc_lon) * p)) / 2
        dist.append(12742 * asin(sqrt(hav)))
    minidx = np.argmin(dist)
    return name[minidx] +'_'+ file

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

            polluted_df['Energy_Loss'] = ((polluted_df['Pmax']*10))#- (pristine_df['Pmax']*10))/(pristine_df['Power'] * 1e9) * 100
            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data

def calc_multiyear_power(locations, start_date, end_date):
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:

            pristine_dir = 'PERC_'+location+'_pristine_'+str(year)
            polluted_dir = 'PERC_'+location+'_'+str(year)

            pristine_df = split_date(fetch_df(pristine_dir))
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = pristine_df['Pmax'] * 10
            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data

def spacial(dir, start_date, end_date, step, locs):

    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)


    locs_dir = os.path.join(os.path.dirname(os.getcwd()),'Location Lists',locs+'.csv')
    locs = pd.read_csv(locs_dir)
    print(locs)
    locs_cap = locs['Nameplate Capacity (MW)'].to_numpy()
    locs_cap_sum = np.sum(locs_cap)
    locs_names = locs['Name'].to_numpy()[::-1]
    locs_lat = locs['Latitude'].to_numpy()
    locs_lon = locs['Longitude'].to_numpy()

    dir = os.path.join(os.path.dirname(os.getcwd()), 'Location Lists', dir + '.csv')
    subfiles = pd.read_csv(dir)
    subfiles_names = [calc_distance(locs_lat[i],locs_lon[i],'California') for i in range(len(subfiles))]

    dates_of_data,energy_loss = calc_multiyear(subfiles_names,start_date,end_date)
    d,power = calc_multiyear_power(subfiles_names,start_date,end_date)

    epoch = dt.datetime(1970,1,1)
    start_date_epoch = (start_date - epoch).total_seconds()
    end_date_epoch = (end_date - epoch).total_seconds()
    duration_seconds = end_date_epoch - start_date_epoch
    duration_minutes = duration_seconds/60
    duration_hours = duration_minutes/60
    duration_days = duration_hours/24

    latitudes = subfiles['Latitude']
    longitudes = subfiles['Longitude']

    latitudes_unique = latitudes.unique()
    longitudes_unique = longitudes.unique()

    locs_lat_arg = [np.abs(latitudes_unique - lat).argmin() for lat in locs_lat][::-1]
    locs_lon_arg = [np.abs(longitudes_unique - lon).argmin() for lon in locs_lon][::-1]


    data = energy_loss
    power_cal = power
    
    dates = pd.date_range(start_date,end_date,freq=str(step)+'H').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list,dates)

    plt.rcParams["figure.figsize"] = (11, 7)
    #fig, ax = plt.subplots(1,1)
    #colours = ['tab:blue','tab:orange','tab:green','tab:red'][::-1]
    for i in range(len(locs_names)):
        data[i] = data[i] * locs_cap[i]
        power_cal[i] = power[i] * locs_cap[i]

    data = np.asarray(data)
    data = np.sum(data,axis=0)/locs_cap_sum
    data = data[:2921]

    power_cal = np.asarray(power_cal)
    power_cal = np.sum(power_cal,axis=0)/locs_cap_sum
    power_cal = power_cal[:2921]

    plt.plot(dates[4::8],data[4::8])
    plt.ylabel('Geographicaly Weighted Power Generation (Wm$^{-1}$)')
    plt.plot(dates[4::8],power_cal[4::8],c='tab:orange')
    plt.xlim(left=dates[0],right=dates[-1])
    return
 
spacial('California_SolarFarms','1/1/2020','31/12/2020','3','California_SolarFarms')
plt.savefig('California_Engery_Loss_SolarFarms.png',dpi=600)
#plt.savefig('California_Engery_Loss.svg')
