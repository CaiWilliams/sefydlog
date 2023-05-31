import cartopy.crs as ccrs
import cartopy.util as ccut
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.patches as mpatches
import cartopy.io.img_tiles as cimgt
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




def spacial_Long(dir, start_date, end_date, step, lat_min, lat_max):

    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)

    dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Location Lists',dir+'.csv')
    subfiles = pd.read_csv(dir)
    subfiles = subfiles.loc[((subfiles['Latitude'] >= lat_min) & (subfiles['Latitude'] <= lat_max))]
    subfiles = subfiles.reset_index()
    subfiles_names = [subfiles.loc[i]['Name'] +'_'+ subfiles.loc[i]['State'] for i in range(len(subfiles))]

    dates_of_data,energy_loss = calc_multiyear(subfiles_names,start_date,end_date,40)
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

    data = np.mean(data[:,:,dates_idx],axis=2)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return longitudes_unique,mean,std




fig = plt.figure(figsize=(5,4))
gs = fig.add_gridspec(20, 20,hspace=0.05,wspace=0.05)

stamen_terrain = cimgt.Stamen('terrain-background')
ax0 = fig.add_subplot(gs[0:10,:], projection=stamen_terrain.crs)
ax0.set_extent([-100, 0, 10, 25], crs=ccrs.PlateCarree())
ax0.add_image(stamen_terrain, 7)
ax0.axhspan(ymin=10,ymax=15,xmin=-100,xmax=0,facecolor='tab:blue',alpha=0.25)
ax0.axhspan(ymin=15,ymax=20,xmin=-100,xmax=0,facecolor='tab:orange',alpha=0.25)
ax0.axhspan(ymin=20,ymax=25,xmin=-100,xmax=0,facecolor='tab:green',alpha=0.25)

ax0.add_patch(mpatches.Rectangle(xy=[0,25],width=-100,height=-5,facecolor='tab:blue',alpha=0.25,transform=ccrs.PlateCarree()))
ax0.add_patch(mpatches.Rectangle(xy=[0,20],width=-100,height=-5,facecolor='tab:orange',alpha=0.25,transform=ccrs.PlateCarree()))
ax0.add_patch(mpatches.Rectangle(xy=[0,15],width=-100,height=-5,facecolor='tab:green',alpha=0.25,transform=ccrs.PlateCarree()))

ax1 = fig.add_subplot(gs[10:,:])

longitudes_unique,mean_lon,std_lon = spacial_Long('Caribbean','12/06/2020','24/06/2020',24,10,15)
ax1.plot(longitudes_unique, mean_lon)

#ax1.fill_between(longitudes_unique, mean_lon - std_lon, mean_lon + std_lon, alpha=0.25)

longitudes_unique,mean_lon,std_lon = spacial_Long('Caribbean','12/06/2020','24/06/2020',24,15,20)
ax1.plot(longitudes_unique, mean_lon)

#ax1.fill_between(longitudes_unique, mean_lon - std_lon, mean_lon + std_lon, alpha=0.25)

longitudes_unique,mean_lon,std_lon = spacial_Long('Caribbean','12/06/2020','24/06/2020',24,20,25)
ax1.plot(longitudes_unique, mean_lon)
ax1.set_xlim(left=-100,right=0)
#ax1.fill_between(longitudes_unique, mean_lon - std_lon, mean_lon + std_lon, alpha=0.25)


ax1.set_ylim(top=-5, bottom=-40)
ax1.set_ylabel('Electrical Power Loss (Wm$^{-2}$)')
ax1.set_xlabel('Longitude ($\degree$)')

plt.tight_layout()
#spacial('EuropeNAmerica_LongitudeLine','12/06/2020','24/06/2020',24)
plt.savefig('Figure_8.png',dpi=600)