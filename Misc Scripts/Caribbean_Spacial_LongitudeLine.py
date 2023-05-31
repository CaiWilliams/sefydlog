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

def spacial_Lat(dir, start_date, end_date, step,lon_min,lon_max):
    start_date = pd.to_datetime(start_date, dayfirst=True, )
    end_date = pd.to_datetime(end_date, dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)

    dir = os.path.join(os.path.dirname(os.getcwd()), 'Location Lists', dir + '.csv')
    subfiles = pd.read_csv(dir)
    subfiles = subfiles.loc[((subfiles['Longitude'] >= lon_min) & (subfiles['Longitude'] <= lon_max))]
    subfiles = subfiles.reset_index()
    subfiles_names = [subfiles.loc[i]['Name'] + '_' + subfiles.loc[i]['State'] for i in range(len(subfiles))]

    dates_of_data, energy_loss = calc_multiyear(subfiles_names, start_date, end_date, 40)
    epoch = dt.datetime(1970, 1, 1)
    start_date_epoch = (start_date - epoch).total_seconds()
    end_date_epoch = (end_date - epoch).total_seconds()
    duration_seconds = end_date_epoch - start_date_epoch
    duration_minutes = duration_seconds / 60
    duration_hours = duration_minutes / 60
    duration_days = duration_hours / 24

    # fig,ax = plt.subplots()

    latitudes = subfiles['Latitude']
    longitudes = subfiles['Longitude']

    latitudes_unique = latitudes.unique()
    longitudes_unique = longitudes.unique()
    longitudes_unique = np.sort(longitudes_unique)

    latitudes = latitudes.values.ravel().reshape((len(latitudes_unique), len(longitudes_unique)))
    longitudes = longitudes.values.ravel().reshape((len(latitudes_unique), len(longitudes_unique)))
    longitudes_unique = subfiles['Longitude'].unique()
    longitudes_unique.sort()
    data = np.zeros((len(latitudes_unique), len(longitudes_unique), len(dates_of_data)))
    for idx, lon in enumerate(longitudes_unique):
        for jdx, lat in enumerate(latitudes_unique):
            index = subfiles.index[(subfiles['Latitude'] == lat) & (subfiles['Longitude'] == lon)].values[0]
            data[jdx, idx, :] = energy_loss[index]

    dates = pd.date_range(start_date, end_date, freq=str(step) + 'H').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list, dates)

    data = np.mean(data[:, :, dates_idx], axis=2)
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    # ax.plot(latitudes_unique,mean)
    # ax.fill_between(latitudes_unique,mean-std,mean+std,alpha=0.25)
    # ax.set_xlim(left=10,right=60)
    # ax.set_ylabel('Mean Electrical Power Loss (W/Wm$^{-2}$)')
    # ax.set_xlabel('Latitude ($\degree$)')

    return latitudes_unique, mean, std

#longitudes_unique,mean_lon,std_lon = spacial_Long('EuropeNAmerica_LongitudeLine','12/06/2020','24/06/2020',24)
#latitude_unique,mean_lat,std_lat = spacial_Lat('EuropeNAmerica_LatitudeLine','23/05/2008','2/06/2008',24)

fig = plt.figure(figsize=(5,4))
gs = fig.add_gridspec(20, 20,hspace=0.05,wspace=0.05)

stamen_terrain = cimgt.Stamen('terrain-background')
ax0 = fig.add_subplot(gs[:,:15], projection=stamen_terrain.crs)
ax0.set_extent([-20, 20, 10, 60], crs=ccrs.PlateCarree())
#ax0.set_xlim(left=-100,right=0)
#ax0.set_ylim(top=60,bottom=10)
#ax0.add_feature(cfeature.LAND)
#ax0.add_feature(cfeature.OCEAN)
ax0.add_image(stamen_terrain, 7)
ax0.add_patch(mpatches.Rectangle(xy=[-20,10],width=10,height=50,facecolor='tab:blue',alpha=0.25,transform=ccrs.PlateCarree()))
ax0.add_patch(mpatches.Rectangle(xy=[-10,10],width=10,height=50,facecolor='tab:orange',alpha=0.25,transform=ccrs.PlateCarree()))
ax0.add_patch(mpatches.Rectangle(xy=[0,10],width=10,height=50,facecolor='tab:green',alpha=0.25,transform=ccrs.PlateCarree()))
ax0.add_patch(mpatches.Rectangle(xy=[10,10],width=10,height=50,facecolor='tab:red',alpha=0.25,transform=ccrs.PlateCarree()))


ax2 = fig.add_subplot(gs[:, 15:])
ax2.set_ylim(top=60,bottom=10)

latitude_unique,mean_lat,std_lat = spacial_Lat('EuropeNAmerica_LatitudeLine','23/05/2008','2/06/2008',24,-20,-10)
ax2.plot(mean_lat,latitude_unique)
ax2.fill_betweenx(latitude_unique, mean_lat - std_lat, mean_lat + std_lat, alpha=0.25)

latitude_unique,mean_lat,std_lat = spacial_Lat('EuropeNAmerica_LatitudeLine','23/05/2008','2/06/2008',24,-10,0)
ax2.plot(mean_lat,latitude_unique)
ax2.fill_betweenx(latitude_unique, mean_lat - std_lat, mean_lat + std_lat, alpha=0.25)

latitude_unique,mean_lat,std_lat = spacial_Lat('EuropeNAmerica_LatitudeLine','23/05/2008','2/06/2008',24,0,10)
ax2.plot(mean_lat,latitude_unique)
ax2.fill_betweenx(latitude_unique, mean_lat - std_lat, mean_lat + std_lat, alpha=0.25)

latitude_unique,mean_lat,std_lat = spacial_Lat('EuropeNAmerica_LatitudeLine','23/05/2008','2/06/2008',24,10,20)
ax2.plot(mean_lat,latitude_unique)
ax2.fill_betweenx(latitude_unique, mean_lat - std_lat, mean_lat + std_lat, alpha=0.25)



ax2.set_xlabel('Electrical Power Loss (Wm$^{-2}$)')
ax2.set_ylabel('Latitude ($\degree$)')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

#fig.subp
#ax2.set_xlim()
#ax2.fill_between(latitude_unique,mean_lat - std)
plt.tight_layout()
#plt.show()
#spacial('EuropeNAmerica_LongitudeLine','12/06/2020','24/06/2020',24)
plt.show()
#plt.savefig('Saharan_longitudeLine.png',dpi=600)
#plt.savefig('California_Spacial.svg')
 
spacial_Lat('EuropeNAmerica_LongitudeLine','12/06/2020','24/06/2020',24,10,15)
spacial_Lat('EuropeNAmerica_LongitudeLine','12/06/2020','24/06/2020',24,15,20)
spacial_Lat('EuropeNAmerica_LongitudeLine','12/06/2020','24/06/2020',24,20,25)
plt.show()
#plt.savefig('Saharan_longitudeLine.png',dpi=600)
#plt.savefig('California_Spacial.svg')