
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import os
import datetime as dt
from math import cos, asin, sqrt
from scipy import stats
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

def spacial(dir, start_date, end_date, step, locs,noct):

    start_date = pd.to_datetime(start_date, dayfirst=True)
    end_date = pd.to_datetime(end_date, dayfirst=True)

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



    data = energy_loss
    
    dates = pd.date_range(start_date,end_date,freq=str(step)+'H').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list,dates)

    for i in range(len(locs_names)):
        data[i] = data[i] * locs_cap[i]

    data = np.asarray(data)
    data = np.sum(data,axis=0)/locs_cap_sum
    data = data[dates_idx]


    return dates, data

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
    return d

def filter_by_date(gen,start_date,end_date):
    start_date = pd.to_datetime(start_date, dayfirst=True)
    end_date = pd.to_datetime(end_date, dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)
    mask = ((gen.index >= start_date) & (gen.index <= end_date))
    gen = gen.loc[mask]
    gen = gen.resample('3h').mean()
    return gen
def norm(y):
    y_max = np.max(y)
    y_min = np.min(y)
    return np.asarray((y - y_min) / (y_max - y_min)).ravel()


fig, ax = plt.subplots()

start_date = '1/1/2020'
end_date = '31/12/2020'

gen = plot_generation()
gen = filter_by_date(gen,start_date,end_date)
norm_gen_values = norm(gen.to_numpy())

mod_dates, mod_values, = spacial('California_SolarFarms', start_date, end_date, '3', 'California_SolarFarms', 40)
norm_mod_values = norm(mod_values)

slope, intercept, r, p, se = stats.linregress(norm_gen_values,norm_mod_values)



print('r^2: ' + str(r**2))


#ax.scatter(norm_gen_values,norm_gen_values)
#xy = np.linspace(0, 1, 10000)
#ax.plot(xy,xy,c='black')
#plt.show()
