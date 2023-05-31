import difflib

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import cartopy.feature as cf
import cartopy.crs as ccrs
from matplotlib import image
import numpy as np
import itertools
import os
import pandas as pd
import natsort
from shapely.geometry import shape
from shapely.geometry import Point
import datetime as dt
matplotlib.use('TkAgg')



def load(file,paramater,units):
    latlons = os.path.join(os.path.dirname(os.getcwd()),'Location Lists','latlons.csv')
    latlons = pd.read_csv(latlons)
    lats = latlons['latitudes'].dropna().to_numpy()
    lons = latlons['longitudes'].dropna().to_numpy()
    lons = (lons + 180) % 360 - 180
    nfile = file
    file = os.path.join(os.path.dirname(os.getcwd()), 'Location Lists', file+'.csv')
    file = pd.read_csv(file)
    min_lat = np.argwhere(lats == np.min(file['Latitude'].to_numpy())).ravel()[0]
    max_lat = np.argwhere(lats == np.max(file['Latitude'].to_numpy())).ravel()[0]
    min_lon = np.argwhere(lons == np.min(file['Longitude'].to_numpy())).ravel()[0]
    max_lon = np.argwhere(lons == np.max(file['Longitude'].to_numpy())).ravel()[0]
    years = natsort.natsorted(os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'Climate Data')))
    months = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    hours = ['0', '3', '6', '9', '12', '15', '18', '21']
    dirs = np.zeros(len(years) * 365 * len(hours) * 2, dtype='<U150')
    dates = np.zeros(len(years) * 365 * len(hours) * 2, dtype=object)
    i = 0
    for year in years:
        for month in months:
            days = natsort.natsorted(os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'Climate Data', year, month)))
            for day in days:
                for hour in hours:
                    possible_dirs = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'Climate Data', year, month, day, hour))
                    possible_dirs = difflib.get_close_matches(paramater, possible_dirs, cutoff=0)
                    dirs[i] = os.path.join(os.path.dirname(os.getcwd()), 'Climate Data', year, month, day, hour, possible_dirs[0])
                    dates[i] = dt.datetime(int(year), int(month), int(day), int(hour))
                    i = i + 1
    dirs = dirs[dirs != '']
    dates = dates[dates != 0]
    data = np.zeros([len(dirs), np.abs(max_lat - min_lat), np.abs(min_lon - max_lon)])
    for idx, dir in enumerate(dirs):
        mins = float(dir.split('_')[-1][3:-4])
        maxs = float(dir.split('_')[-2][3:])
        img = np.asarray(image.imread(dir))
        img = img[max_lat:min_lat,min_lon:max_lon,0]/255 * 3
        img = img * (maxs-mins) + mins
        data[idx] = img

    data = np.mean(data,axis=(1,2))
    d = pd.DataFrame()
    d.index = pd.DatetimeIndex(dates)
    d['Y'] = data
    d['Y'] = d['Y'].rolling('365d', center=True).mean()
    plt.plot(d.index, d['Y'])
    plt.ylabel(paramater) #+ ' ' + '(' + units + ')')
    plt.xlim(left=dates[0],right=dates[-1])
    p = paramater.replace(' ','_')
    plt.savefig(str(nfile) + '_' + str(p) + '.png',dpi=600)

load('Beijing','Air Temperature','ppm')
load('Beijing','Carbon Mononxide','ppm')
load('Beijing','Formaldehyde','ppm')
load('Beijing','Methane','ppm')
load('Beijing','Nitric Acid','ppm')
load('Beijing','Nitrogen Dioxide','ppm')
load('Beijing','Ozone','ppm')
load('Beijing','Relative Humidity','ppm')
load('Beijing','Sulfur Dioxide','ppm')
load('Beijing','Surface Air Pressure','ppm')
load('Beijing','TAU550','ppm')
load('Beijing','Water Vapour','ppm')