import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp2d
import datetime as dt

def fetch_df(name):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    data = pd.read_csv(file)
    return data

def split_date(df):
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df = df.set_index('Date')
    df.loc[:,'Hour'] = df.index.hour.values
    df.loc[:,'Day'] = df.index.day.values
    df.loc[:,'Month'] = df.index.month.values
    return df

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

            polluted_df['Energy_Loss'] = (polluted_df['Pmax']*10) - (pristine_df['Pmax']*10)
            polluted.append(polluted_df)
        polluted = pd.concat(polluted)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data


def plot_from_file_spacial(dir,start_year,end_year):
    dir = os.path.join(os.getcwd(),dir+'.csv')
    data = pd.read_csv(dir)
    names = []
    for i in range(len(data)):
        name = str(data.loc[i]['Name']) + '_' + str(data.loc[i]['State'])
        names.append(name)
    dates,data_z = calc_multiyear(names,start_year,end_year)
    z = np.zeros((14,14,2920))
    unique_lat = data['Latitude'].unique()
    unique_lon = data['Longitude'].unique()

    for idx,lat in enumerate(unique_lat):
        for jdx,lon in enumerate(unique_lon):
            index = data.index[(data['Latitude'] == lat)&(data['Longitude'] == lon)].values[0]
            z[jdx,idx,:] = data_z[index]
    d = dates.values
    d_filter = dates[(dates.hour == 12) & (((dates.day >= 14) & (dates.month == 8)) | ((dates.day <= 10) & (dates.month == 9)))].values
    d = np.searchsorted(d,d_filter)
    plt.rcParams["figure.figsize"] = (11,7)
    fig, ax = plt.subplots(ncols=7,nrows=4)
    x = 0
    y = 0
    for i in d:
        im = ax[y,x].imshow(z[:,:,i], cmap='inferno_r',vmin=-100,vmax=0)
        ax[y,x].set_axis_off()
        ax[y,x].title.set_text(str(dates[i].day)+'-'+str(dates[i].month)+'-'+str(dates[i].year))
        x += 1
        if x >= 7:
            x = 0
            y += 1
    cbar_ax = fig.add_axes([0.905, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax,label='Energy Lost (Wm$^{-2}$)')
    dir = os.path.join(os.getcwd(),'Figures','California_2020_Spectral_NO_Fire')
    return


def spacial(dir, start_date, end_date, step):

    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)

    dir = os.path.join(os.getcwd(),dir+'.csv')
    subfiles = pd.read_csv(dir)
    subfiles_names = [subfiles.loc[i]['Name'] +'_'+ subfiles.loc[i]['State'] for i in range(len(subfiles))]
    
    dates_of_data,energy_loss = calc_multiyear(subfiles_names,start_date,end_date)
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
        fig,ax = plt.subplots(ncols=7,nrows=int(duration_hours/step/7)+1,subplot_kw={'projection': ccrs.PlateCarree()})

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
            data[idx,jdx,:] = energy_loss[index]

    shapename = 'admin_1_states_provinces_lakes'
    states_shp = shpreader.natural_earth(resolution='110m',
                                         category='cultural', name=shapename)

    dates = pd.date_range(start_date,end_date,freq=str(step)+'H').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list,dates)
    ax = ax.flatten()
    for i in range(len(ax)):
        if i >= len(dates_idx):
            fig.delaxes(ax[i])
        else:
            l = np.around(np.arange(-100,-5,5),0)
            cs = ax[i].pcolormesh(longitudes,latitudes,data[:,:,dates_idx[i]],transform = ccrs.PlateCarree(),cmap='inferno_r')
            year = str(dates_of_data[dates_idx[i]].year)
            month = str(dates_of_data[dates_idx[i]].month)
            day = str(dates_of_data[dates_idx[i]].day)
            ax[i].set_title(year+'-'+month+'-'+day,fontsize=10)
            ax[i].add_feature(cfeature.STATES.with_scale('110m'),linestyle=':',edgecolor='white')
            ax[i].coastlines(color='white')
            #ax[i].plot(-118.5426,34.3917,transform = ccrs.PlateCarree(),marker='o',markerfacecolor='none',markeredgewidth=1,markersize=3)
            #ax[i].plot(-119.7871,36.7378,transform = ccrs.PlateCarree(),marker='o',markerfacecolor='none',markeredgewidth=1,markersize=3)
            #ax[i].plot(-121.4911,38.5816,transform = ccrs.PlateCarree(),marker='o',markerfacecolor='none',markeredgewidth=1,markersize=3)
            #ax[i].plot(-122.3917,40.5865,transform = ccrs.PlateCarree(),marker='o',markerfacecolor='none',markeredgewidth=1,markersize=3)
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.05, right=0.95, wspace=0.05, hspace=0.05)
    cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.02])
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',label='Energy Lost (Wm$^{-2}$)')
    return
 
spacial('BritishIsles1','16/08/2020','25/09/2020',24)
plt.show()
#plt.savefig('California_Spacial.png',dpi=1200)
#plt.savefig('California_Spacial.svg')