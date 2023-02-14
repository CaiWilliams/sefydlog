import os.path

import pandas as pd
import numpy as np
import itertools  
import secrets     

def generate(lat_min,lat_max,lon_min,lon_max,name):
    data = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'Location Lists','latlons.csv'))
    latitudes = data['latitudes'].dropna(axis=0).to_numpy()
    longitudes = data['longitudes'].to_numpy()
    longitudes = (longitudes + 180) % 360 - 180

    num_lat = np.where((latitudes > lat_min) & (latitudes <= lat_max),latitudes,np.nan)
    num_lat = num_lat[~np.isnan(num_lat)][::2]

    num_lon = np.where((longitudes > lon_min) & (longitudes  <= lon_max),longitudes,np.nan)
    num_lon = num_lon[~np.isnan(num_lon)][::2]

    lat_lons = np.asarray(list(itertools.product(num_lat,num_lon)))
    lats = lat_lons[:,0]
    lons = lat_lons[:,1]
    names = [str(secrets.token_hex(8)) for x in range(len(lats))]

    data = pd.DataFrame(index=None)
    data['Name'] = names
    data['State'] = str(name)
    data['Latitude'] = lats
    data['Longitude'] = lons

    data.to_csv(str(name)+'.csv', index=False)
    return


#generate(18.75,42,98.25,123,'China_LowRes')