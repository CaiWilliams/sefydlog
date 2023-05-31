
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
            #pristine_dir = 'PERC_' + location + '_pristine_' + str(year)
            #polluted_dir = 'PERC_' + location + '_' + str(year)
            #pristine_dir = 'PERC_' + location + '_pristine_tilt_angle__NOCT_'+ str(noct) + '_' + str(year)
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)

            #pristine_df = split_date(fetch_df(pristine_dir))
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
            #pristine_dir = 'PERC_' + location + '_pristine_' + str(year)
            #polluted_dir = 'PERC_' + location + '_' + str(year)
            pristine_dir = 'PERC_' + location + '_pristine_tilt_angle__NOCT_'+ str(noct) + '_' + str(year)
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)

            pristine_df = split_date(fetch_df(pristine_dir))
            polluted_df = split_date(fetch_df(polluted_dir))

            #polluted_df['Energy_Loss'] = ((polluted_df['Pmax'] * 10)) # - (pristine_df['Pmax']*10))/(pristine_df['Power'] * 1e9) * 100
            polluted_df['Energy_Loss'] = pristine_df['Pmax'] * 10
            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data

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
    data = data[dates_idx]

    power_cal = np.asarray(power_cal)
    power_cal = np.sum(power_cal,axis=0)/locs_cap_sum
    power_cal = power_cal[dates_idx]

    # ax.plot(dates,data,c='tab:olive')
    # #plt.ylabel('Geographicaly Weighted Power Generation (Wm$^{-2}$)')
    # ax.plot(dates,power_cal,c='tab:cyan')
    # ax.set_ylim(bottom=0)
    #plt.xlim(left=dates[0],right=dates[-1])
    return dates, data, power_cal

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
    fires_start = dt.datetime(year=2020,month=8,day=16,hour=12)
    fires_end = dt.datetime(year=2020,month=9,day=25,hour=12)
    plt.xlim(d.iloc[0].name,d.iloc[-1].name)
    dates = [dt.datetime(year=2020,month=1,day=1,hour=0),dt.datetime(year=2020,month=4,day=1,hour=0),dt.datetime(year=2020,month=7,day=1,hour=0),dt.datetime(year=2020,month=10,day=1,hour=0),dt.datetime(year=2021,month=1,day=1,hour=0)]
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
    data = data.resample('1h').mean()
    temp = pd.concat([gen,data]).sort_index()
    temp = temp.groupby(temp.index).sum()
    temp['No Cut'] = temp['Solar Curtailment'] + temp['MW'].clip(lower=0)
    temp = temp.resample('24h').mean()
    #temp.index = temp.index.duplicated(keep='first')
    x = temp.index.to_numpy()
    #y = temp['Solar Curtailment'].to_numpy()
    y1 = temp['MW'].clip(lower=0).to_numpy()
    return x,y1
    # ax.stackplot(x,y1,color='tab:gray')
    # axins = ax.inset_axes([0.5, 0.5, 100, 160])
    # axins.stackplot(x,y1,color='tab:gray')
    # ax.set_ylabel('Mean Daily Generation from Solar Assets (MW)')

def reduction(date,peak_date,x,y):
    date = pd.to_datetime(date,dayfirst=True)
    peak_date = pd.to_datetime(peak_date,dayfirst=True)
    try:
        peak_idx = np.argwhere(x == peak_date).ravel()[0]
        date_idx = np.argwhere(x == date).ravel()[0]
    except:
        date = date.replace(hour=12)
        peak_date = peak_date.replace(hour=12)
        peak_idx = np.argwhere(x == peak_date).ravel()[0]
        date_idx = np.argwhere(x == date).ravel()[0]
    prev7 = date_idx - 7
    prev_mean = np.mean(y[prev7:date_idx])
    #print(prev_mean)
    date_value = y[peak_idx]
    #print(date_value)
    change = ((prev_mean - date_value)/prev_mean) * 100
    print(change)

date = '04/09/2020'
peak_date = '11/09/2020'
fig, ax  = plt.subplots()
gen = plot_generation()
x,y = plot_curtailment(gen)
reduction(date,peak_date,x,y)

ax.stackplot(x,y,color='tab:gray')
ax.set_ylabel('Mean Daily Generation from Solar Assets (MW)')
ax2 = ax.twinx()
dates, data, power_cal = spacial('California_SolarFarms','1/6/2020','30/10/2020','24','California_SolarFarms',40)

reduction(date,peak_date,dates,data)

ax2.plot(dates,data,c='tab:olive')
ax2.plot(dates,power_cal,c='tab:cyan')
ax2.set_ylabel('Geographicaly Weighted Power Generation (Wm$^{-2}$)')
ax2.set_ylim(bottom=0)


#plt.show()
plt.show()
#plt.savefig('California_Engery_Loss_SolarFarms.png',dpi=600)
#plt.savefig('California_Engery_Loss_SolarFarms_cloud_cover.png',dpi=600)
#plt.savefig('California_Engery_Loss.svg')
