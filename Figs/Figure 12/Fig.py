import cartopy.crs as ccrs
import cartopy.util as ccut
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp2d
import datetime as dt

def fetch_df(name):
    file = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Results',name+'.csv')
    data = pd.read_csv(file)
    return data

def split_date(df):
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df = df.set_index('Date')
    df.loc[:,'Hour'] = df.index.hour.values
    df.loc[:,'Day'] = df.index.day.values
    df.loc[:,'Month'] = df.index.month.values
    return df

def calc_multiyear(locations, start_date, end_date,noct):
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:

            #pristine_dir = 'PERC_'+location+'_pristine_'+str(year)
            #polluted_dir = 'PERC_'+location+'_'+str(year)

            pristine_dir = 'FullRes_PERC_' + location + '_pristine_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            polluted_dir = 'FullRes_PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)

            pristine_df = split_date(fetch_df(pristine_dir))
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = (((polluted_df['Pmax']*10)) - (pristine_df['Pmax']*10)) #/ (pristine_df['Power'] * 1e9) * 100
            polluted_df['Energy_Loss'] = polluted_df['Energy_Loss'].clip(upper=0)

            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data

def running_mean(x, N):
   cumsum = np.cumsum(np.insert(x, 0, 0))
   return (cumsum[N:] - cumsum[:-N]) / N


def spacial(dir, start_date, end_date, step):

    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)

    dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Location Lists',dir+'.csv')
    subfiles = pd.read_csv(dir)
    subfiles_names = [subfiles.loc[i]['Name'] +'_'+ subfiles.loc[i]['State'] for i in range(len(subfiles))]
    
    dates_of_data,energy_loss = calc_multiyear(subfiles_names,start_date,end_date,40)
    epoch = dt.datetime(1970,1,1)
    start_date_epoch = (start_date - epoch).total_seconds()
    end_date_epoch = (end_date - epoch).total_seconds()
    duration_seconds = end_date_epoch - start_date_epoch
    duration_minutes = duration_seconds/60
    duration_hours = duration_minutes/60
    duration_days = duration_hours/24

    fig,ax = plt.subplots()


    latitudes = subfiles['Latitude']
    longitudes = subfiles['Longitude']

    latitudes_unique = latitudes.unique()
    longitudes_unique = longitudes.unique()
    longitudes_unique = np.sort(longitudes_unique)


    latitudes = latitudes.values.ravel().reshape((len(latitudes_unique),len(longitudes_unique)))
    longitudes = longitudes.values.ravel().reshape((len(latitudes_unique),len(longitudes_unique)))
    longitudes_unique = subfiles['Longitude'].unique()
    longitudes_unique.sort()
    data = np.zeros((len(latitudes_unique),len(longitudes_unique),len(dates_of_data)))
    for idx,lon in enumerate(longitudes_unique):
        for jdx,lat in enumerate(latitudes_unique):
            index = subfiles.index[(subfiles['Latitude'] == lat)&(subfiles['Longitude'] == lon)].values[0]
            data[jdx,idx,:] = energy_loss[index]

    dates = pd.date_range(start_date,end_date,freq=str(step)+'H').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list,dates)

    loss = np.zeros(len(dates_idx))
    for i in range(len(dates_idx)):
        loss[i] = np.mean(data[:,:,dates_idx[i]])
        #loss[i] = np.percentile(data[:,:,dates_idx[i]],30)
    data = pd.DataFrame()
    data['dates'] = dates
    data['loss'] = loss
    data = data.set_index('dates')
    #ax = plt.plot(data.index,data['loss'])
    data['loss'] = data['loss'].rolling('365d',center=True).mean()
    print(data['loss'].iloc[0])
    print(data['loss'].iloc[-1])
    #data['loss'] = ((data['loss'].iloc[0]/data['loss']) - 1) * 100
    ax = plt.plot(data.index,(data['loss']-data['loss'][0])/data['loss'][0])
    #plt.ylabel('Electrical Power Loss Reduction (%)')
    plt.axvline(dt.datetime(2014,1,1,0,0),color='tab:orange',linestyle='--')
    plt.ylabel('Electrical Power Loss Rolling Mean (1 Year) (Wm$^{-2}$)')
    plt.xlim(left=dt.datetime(2003,1,1,0,0),right=dt.datetime(2020,12,31,0,0))
    #ax = plt.axvspan(dt.datetime(2008,8,8,0,0),dt.datetime(2008,8,24,0,0),color='tab:green',alpha=0.25)
    return
 
spacial('Beijing','1/01/2014','31/12/2020',24)
#plt.savefig('Figure_12.png',dpi=600)
plt.show()
#plt.savefig('Caribbean_Spacial_2020.png',dpi=600)
#plt.savefig('California_Spacial.svg')