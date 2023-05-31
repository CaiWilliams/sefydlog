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
            pristine_dir = 'PERC_' + location + '_pristine_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            #pristine_dir = 'PERC_'+location+'_pristine_'+str(year)
            #polluted_dir = 'PERC_'+location+'_'+str(year)

            pristine_df = split_date(fetch_df(pristine_dir)) 
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = (((polluted_df['Pmax']*10)) - (pristine_df['Pmax']*10)) #/ (pristine_df['Power'] * 1e9) * 100

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
    
    dates_of_data,energy_loss = calc_multiyear(subfiles_names,start_date,end_date,40)
    epoch = dt.datetime(1970,1,1)
    start_date_epoch = (start_date - epoch).total_seconds()
    end_date_epoch = (end_date - epoch).total_seconds()
    duration_seconds = end_date_epoch - start_date_epoch
    duration_minutes = duration_seconds/60
    duration_hours = duration_minutes/60
    duration_days = duration_hours/24
    
    if duration_hours/step < 7:
        fig,ax = plt.subplots(ncols=int(duration_hours/step)+1,nrows=1,subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig,ax = plt.subplots(ncols=4,nrows=2, subplot_kw={'projection': ccrs.PlateCarree()})
        fig.set_size_inches(6,3)

    latitudes = subfiles['Latitude']
    longitudes = subfiles['Longitude']

    latitudes_unique = latitudes.unique()
    longitudes_unique = longitudes.unique()

    latitudes = latitudes.values.ravel().reshape((len(longitudes_unique),len(latitudes_unique)))
    longitudes = longitudes.values.ravel().reshape((len(longitudes_unique),len(latitudes_unique)))


    data = np.zeros((len(longitudes_unique),len(latitudes_unique),len(dates_of_data)))
    for idx,lon in enumerate(longitudes_unique):
        for jdx,lat in enumerate(latitudes_unique):
            index = subfiles.index[(subfiles['Latitude'] == lat)&(subfiles['Longitude'] == lon)].values[0]
            data[jdx,idx,:] = energy_loss[index]

    dates = pd.date_range(start_date,end_date,freq=str(step)+'H').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list,dates)
    ax = ax.flatten()
    test = data[:, :, dates_idx]
    print(np.min(test))
    for i in range(len(ax)):
        if i >= len(dates_idx):
            fig.delaxes(ax[i])
        else:
            l = np.around(np.arange(-100,2,2),0)
            cs = ax[i].contourf(longitudes,latitudes,data[:,:,dates_idx[i]],transform = ccrs.PlateCarree(),levels=l,cmap='inferno_r')
            if i != 0 and i < 4:
                ax[i].set_title('+'+str(int(step*i/24))+' Days', fontsize=9, pad=3)
            elif i >= 4 and i < len(dates_idx)-1:
                ax[i].text(x=0.5, y=-0.15, s='+'+str(int(step*i/24))+' Days', transform=ax[i].transAxes, fontsize=9,ha='center')
            ax[i].add_feature(cfeature.STATES.with_scale('110m'), linestyle=':', edgecolor='white')
            ax[i].coastlines(color='white')
            ax[i].plot(-118.5426, 34.3917, transform=ccrs.PlateCarree(), marker='o', markersize=2)
            ax[i].plot(-119.7871, 36.7378, transform=ccrs.PlateCarree(), marker='o', markersize=2)
            ax[i].plot(-121.4911, 38.5816, transform=ccrs.PlateCarree(), marker='o', markersize=2)
            ax[i].plot(-122.3917, 40.5865, transform=ccrs.PlateCarree(), marker='o', markersize=2)

    fig.subplots_adjust(bottom=0.20, top=0.80, left=0.05, right=0.82, wspace=0.01, hspace=0.05)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='vertical',label='Electrical Power Loss (Wm$^{-2}$)', ticks=[-0,-20,-40,-60,-80,-100])
    fig.text(0.05, 0.65, '2020-9-5', va='center', rotation='vertical', fontsize=9)
    fig.text(0.8,0.35,'2020-9-19', va='center', rotation='vertical',fontsize=9)
    return
 
spacial('California','5/09/2020','19/09/2020',48)
#plt.show()
plt.savefig('California_Spacial.png',dpi=600)
#plt.savefig('California_Spacial.svg')