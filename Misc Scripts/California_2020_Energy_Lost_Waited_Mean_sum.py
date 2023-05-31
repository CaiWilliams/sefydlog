
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import os
import datetime as dt
from math import cos, asin, sqrt

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

def calc_multiyear(locations, start_date, end_date,noct):
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)

            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = polluted_df['Pmax']*10
            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data

def calc_multiyear_power(locations, start_date, end_date,noct):
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:
            pristine_dir = 'PERC_' + location + '_pristine_tilt_angle__NOCT_'+ str(noct) + '_' + str(year)
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)

            pristine_df = split_date(fetch_df(pristine_dir))
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = pristine_df['Pmax'] * 10
            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data

def calc_multiyear_PCE(locations, start_date, end_date,noct):
    years = np.arange(start_date.year,end_date.year+1)
    data_polluted = []
    data_pristine = []
    for location in locations:
        polluted = []
        pristine = []
        for year in years:
            pristine_dir = 'PERC_' + location + '_pristine_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)

            polluted_df = split_date(fetch_df(polluted_dir))
            pristine_df = split_date(fetch_df(pristine_dir))
            
            polluted.append(polluted_df)
            pristine.append(pristine_df)
        polluted = pd.concat(polluted).fillna(0)
        pristine = pd.concat(pristine).fillna(0)
        data_polluted.append(polluted['PCE'].to_numpy())
        data_pristine.append(pristine['PCE'].to_numpy())
    polluted.sort_index()
    pristine.sort_index()
    return polluted.index, data_polluted, data_pristine

def spacial(dir, start_date, end_date, step, locs,noct):

    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)


    locs_dir = os.path.join(os.path.dirname(os.getcwd()),'Location Lists',locs+'.csv')
    locs = pd.read_csv(locs_dir)
    locs_cap = locs['Nameplate Capacity (MW)'].to_numpy()
    locs_cap_sum = np.sum(locs_cap)
    locs_names = locs['Name'].to_numpy()[::-1]
    locs_lat = locs['Latitude'].to_numpy()
    locs_lon = locs['Longitude'].to_numpy()

    dir = os.path.join(os.path.dirname(os.getcwd()), 'Location Lists', dir + '.csv')
    subfiles = pd.read_csv(dir)
    subfiles_names = [calc_distance(locs_lat[i],locs_lon[i],'California') for i in range(len(subfiles))]

    dates_of_data,energy_loss = calc_multiyear(subfiles_names,start_date,end_date,noct)
    d,power = calc_multiyear_power(subfiles_names,start_date,end_date,noct)
    d,polluted_pce,pristine_pce = calc_multiyear_PCE(subfiles_names,start_date,end_date,noct)

    data = energy_loss
    power_cal = power
    
    dates = pd.date_range(start_date,end_date,freq=str(step)+'H').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list,dates)

    for i in range(len(locs_names)):
        data[i] = data[i] * locs_cap[i]
        power_cal[i] = power[i] * locs_cap[i]
        polluted_pce[i] = polluted_pce[i] * locs_cap[i]
        pristine_pce[i] = pristine_pce[i] * locs_cap[i]

    data = np.asarray(data)
    data = np.sum(data, axis=0)/locs_cap_sum
    data = data[dates_idx]
    data = np.interp(np.linspace(0,1,len(data)*3),np.linspace(0,1,len(data)),data)

    power_cal = np.asarray(power_cal)
    power_cal = np.sum(power_cal, axis=0)/locs_cap_sum
    power_cal = power_cal[dates_idx]
    power_cal = np.interp(np.linspace(0,1,len(power_cal)*3),np.linspace(0,1,len(power_cal)),power_cal)

    polluted_pce = np.asarray(polluted_pce)
    polluted_pce = np.sum(polluted_pce, axis=0)/locs_cap_sum
    polluted_pce = polluted_pce[dates_idx]
    polluted_pce = np.interp(np.linspace(0,1,len(polluted_pce)*3),np.linspace(0,1,len(polluted_pce)),polluted_pce)

    pristine_pce = np.asarray(pristine_pce)
    pristine_pce = np.sum(pristine_pce, axis=0) / locs_cap_sum
    pristine_pce = pristine_pce[dates_idx]
    pristine_pce = np.interp(np.linspace(0, 1, len(pristine_pce) * 3), np.linspace(0, 1, len(pristine_pce)), pristine_pce)

    return dates, data, power_cal,polluted_pce,pristine_pce, locs_cap_sum








dates, polluted, pristine, polluted_PCE, pristine_PCE, total_capacity = spacial('California_SolarFarms','16/8/2020','12/11/2020','3','California_SolarFarms',40)

pristine = np.sum(pristine)
polluted = np.sum(polluted)

no_prist_pce = np.nonzero(pristine_PCE)
pristine_PCE = pristine_PCE[no_prist_pce]

no_poll_pce = np.nonzero(polluted_PCE)
polluted_PCE = polluted_PCE[no_poll_pce]

total_capacity = total_capacity * 1e3
loss = (pristine-polluted)/1e3
print(total_capacity/loss)
print('Energy Loss: ' + str(loss) + ' kWh/m^2')
print('Pristine PCE: ' + str(np.mean(pristine_PCE)*100) + ' %')
print('Polluted PCE: ' + str(np.mean(polluted_PCE)*100) + ' %')