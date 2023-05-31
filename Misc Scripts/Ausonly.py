import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import cartopy.feature as cf
import cartopy.crs as ccrs
import numpy as np
import itertools
import os
import pandas as pd
from shapely.geometry import shape
from shapely.geometry import Point
import datetime as dt
matplotlib.use('TkAgg')

def find_nearest(shape,point_array,file):
    df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'Location Lists', file + '.csv'))
    name = df['Name'].to_numpy()
    lat = df['Latitude'].to_numpy()
    lon = df['Longitude'].to_numpy()
    cap = df['Nameplate Capacity (MW)'].to_numpy()
    point_array = np.asarray(point_array)
    dist = []

    for i in range(len(lat)):
        p = 0.017453292519943295
        hav = 0.5 - np.cos((lat[i] - point_array[:,1]) * p) / 2 + np.cos(point_array[:,1] * p) * np.cos(lat[i] * p) * (
                    1 - np.cos((lon[i] - point_array[:,0]) * p)) / 2
        dist.append(np.argmin(12742 * np.arcsin(np.sqrt(hav))))

    dist_arg = dist
    dist = np.asarray([point_array[idx] for idx in dist])
    capacity = np.zeros(len(point_array))


    for jdx,idx in enumerate(dist_arg):
        capacity[idx] = capacity[idx] + cap[jdx]

    capacity = capacity.reshape((shape[0],shape[1]))


    return capacity.T





def fetch_df(name):
    file = os.path.join(os.path.dirname(os.getcwd()), 'Results', name + '.csv')
    data = pd.read_csv(file)
    return data


def split_date(df):
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df = df.set_index('Date')
    df.loc[:, 'Hour'] = df.index.hour.values
    df.loc[:, 'Day'] = df.index.day.values
    df.loc[:, 'Month'] = df.index.month.values
    return df


def calc_multiyear(locations, start_date, end_date,noct):
    years = np.arange(start_date.year, end_date.year + 1)
    data = []
    for location in locations:
        polluted = []
        for year in years:
            pristine_dir = 'PERC_' + location + '_pristine_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            #pristine_dir = 'PERC_' + location + '_pristine_' + str(year)
            #polluted_dir = 'PERC_' + location + '_' + str(year)

            pristine_df = split_date(fetch_df(pristine_dir))
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = (((polluted_df['Pmax'] * 10)) - (pristine_df['Pmax'] * 10)) / (
                        pristine_df['Power'] * 1e9) * 100

            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data


def spacial(dir, start_date, end_date, step):
    start_date = pd.to_datetime(start_date, dayfirst=True, )
    end_date = pd.to_datetime(end_date, dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)

    dir = os.path.join(os.path.dirname(os.getcwd()), 'Location Lists', dir + '.csv')
    subfiles = pd.read_csv(dir)
    subfiles_names = [subfiles.loc[i]['Name'] + '_' + subfiles.loc[i]['State'] for i in range(len(subfiles))]

    dates_of_data, energy_loss = calc_multiyear(subfiles_names, start_date, end_date,20)
    epoch = dt.datetime(1970, 1, 1)
    start_date_epoch = (start_date - epoch).total_seconds()
    end_date_epoch = (end_date - epoch).total_seconds()
    duration_seconds = end_date_epoch - start_date_epoch
    duration_minutes = duration_seconds / 60
    duration_hours = duration_minutes / 60
    duration_days = duration_hours / 24

    latitudes = subfiles['Latitude']
    longitudes = subfiles['Longitude']

    latitudes_unique = latitudes.unique()
    longitudes_unique = longitudes.unique()

    data = np.zeros((len(latitudes_unique), len(longitudes_unique), len(dates_of_data)))
    print(np.shape(data))
    for idx, lon in enumerate(longitudes_unique):
        for jdx, lat in enumerate(latitudes_unique):
            index = subfiles.index[(subfiles['Latitude'] == lat) & (subfiles['Longitude'] == lon)].values[0]
            data[jdx, idx, :] = energy_loss[index]

    dates = pd.date_range(start_date, end_date, freq=str(step) + 'h').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list, dates)
    data = data[:,:,dates_idx]
    lon_lats = list(itertools.product(longitudes_unique,latitudes_unique))
    lon_lats = [list(i) for i in lon_lats]
    return lon_lats, data, dates

def mask_california(lon_lat,X):
    x = cf.STATES.with_scale('10m').geometries()

    for idx, i in enumerate(x):
        if idx == 1257:
            Cali = shape(i.geoms[0])

    lon = np.shape(X)[0]
    lat = np.shape(X)[1]
    step = np.shape(X)[2]

    lon_lat = [Point(x) for x in lon_lat]
    mask = np.asarray([Cali.contains(x) for x in lon_lat])
    mask = mask.reshape((lon, lat)).T
    mask = np.repeat(mask[:,:,np.newaxis], repeats=step, axis=2)
    X = np.where(mask,X,np.nan)
    return X


L, X, dates = spacial('California', '16/08/2020', '12/11/2020', 3)
X = mask_california(L, X)
avg = np.nanmean(X)
std = np.nanstd(X)
test = avg - std
print(test)


area = np.zeros(X.shape[-1])
for idx in range(X.shape[-1]):
    area[idx] = np.sum((np.where(X[:, :, idx] <= test,1,0)))

plt.plot(dates[::], area[::] * 75**2 / 423970)
plt.ylabel('Percent of Californian Land with \n Electrical Power Loss worse than mean - std (%)')
plt.show()
