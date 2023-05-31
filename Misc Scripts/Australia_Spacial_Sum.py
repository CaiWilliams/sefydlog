import cartopy.crs as ccrs
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

def calc_multiyear(locations, start_date, end_date,noct):
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:

            #pristine_dir = 'PERC_'+location+'_pristine_tilt_angle_'+str(year)
            #polluted_dir = 'PERC_'+location+'_tilt_angle_'+str(year)
            pristine_dir = 'PERC_' + location + '_pristine_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)

            pristine_df = split_date(fetch_df(pristine_dir))
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = (((polluted_df['Pmax']*10))) #- (pristine_df['Pmax']*10))#/ (pristine_df['Power'] * 1e9) * 100

            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data

def calc_multiyear2(locations, start_date, end_date,noct):
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:

            #pristine_dir = 'PERC_'+location+'_pristine_tilt_angle_'+str(year)
            #polluted_dir = 'PERC_'+location+'_tilt_angle_'+str(year)
            pristine_dir = 'PERC_' + location + '_pristine_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)

            pristine_df = split_date(fetch_df(pristine_dir))
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = (pristine_df['Pmax'] * 10)

            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data




def spacial(dir, start_date, end_date, step):

    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)

    dir = os.path.join(os.path.dirname(os.getcwd()),'Location Lists',dir+'.csv')
    subfiles = pd.read_csv(dir)
    subfiles_names = [subfiles.loc[i]['Name'] +'_'+ subfiles.loc[i]['State'] for i in range(len(subfiles))]
    
    dates_of_data, energy_loss = calc_multiyear(subfiles_names,start_date,end_date,40)
    dates_of_data, Irradiance = calc_multiyear2(subfiles_names, start_date, end_date, 40)
    epoch = dt.datetime(1970,1,1)
    start_date_epoch = (start_date - epoch).total_seconds()
    end_date_epoch = (end_date - epoch).total_seconds()
    duration_seconds = end_date_epoch - start_date_epoch
    duration_minutes = duration_seconds/60
    duration_hours = duration_minutes/60
    duration_days = duration_hours/24


    fig,ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})


    latitudes = subfiles['Latitude']
    longitudes = subfiles['Longitude']

    latitudes_unique = latitudes.unique()
    longitudes_unique = longitudes.unique()

    latitudes = latitudes.values.ravel().reshape((len(latitudes_unique),len(longitudes_unique)))
    longitudes = longitudes.values.ravel().reshape((len(latitudes_unique),len(longitudes_unique)))


    data = np.zeros((len(latitudes_unique),len(longitudes_unique),len(dates_of_data)*3))
    data2 = np.zeros((len(latitudes_unique), len(longitudes_unique), len(dates_of_data) * 3))
    print(np.shape(data))
    for idx,lon in enumerate(longitudes_unique):
        for jdx,lat in enumerate(latitudes_unique):
            index = subfiles.index[(subfiles['Latitude'] == lat)&(subfiles['Longitude'] == lon)].values[0]
            data[jdx,idx,:] = np.interp(x=np.linspace(0,1,len(energy_loss[index])*3), xp=np.linspace(0,1,len(energy_loss[index])), fp=energy_loss[index])#energy_loss[index]
            data2[jdx, idx, :] = np.interp(x=np.linspace(0, 1, len(Irradiance[index]) * 3), xp=np.linspace(0, 1, len(Irradiance[index])), fp=Irradiance[index])


    #data = np.sum(data,axis=2)
    #data2 = np.sum(data2, axis=2)
    #l = np.arange(0,10.01,0.01)
    cs = ax.contourf(longitudes,latitudes,data2[0],transform = ccrs.PlateCarree(),levels=50,cmap='inferno_r')
    #year = str(dates_of_data[dates_idx[i]].year)
    #month = str(dates_of_data[dates_idx[i]].month)
    #day = str(dates_of_data[dates_idx[i]].day)

    ax.add_feature(cfeature.STATES.with_scale('110m'),linestyle=':',edgecolor='white')
    ax.coastlines(color='white')

    fig.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.82, wspace=0.01, hspace=0.05)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='vertical',label='Percentage Electrical Energy Loss (%)',ticks=[0,2,4,6,8,10])
    return
 
spacial('Australia_LowRes','12/12/2019','12/12/2019',3)
plt.show()
#plt.savefig('Australia_Spacial_Cumulative_2020.png',dpi=600)
#plt.savefig('California_Spacial.svg')