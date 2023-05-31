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
    print(longitudes)

    num_lat = np.where((latitudes >= lat_min) & (latitudes <= lat_max),latitudes,np.nan)
    num_lat = num_lat[~np.isnan(num_lat)]

    num_lon = np.where((longitudes >= lon_min) & (longitudes  <= lon_max),longitudes,np.nan)
    num_lon = num_lon[~np.isnan(num_lon)]


    #lat_lons = np.asarray(list(itertools.product(num_lat,num_lon,repeat=1)))
    lat_lons = np.asarray([[x,y] for x in num_lat for y in num_lon])
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

#generate(-46,-9,112,155,'Australia')
#generate(32,42,-125,-113,'California')
#generate(4,45,-110,1,'Caribbean')
#generate(4,60,-20,20,'EuropeNAmerica_LatitudeLine')
#generate(10,25,-110,1,'EuropeNAmerica_LongitudeLine')
generate(36,43,113,123,'Beijing')
#generate(18.75,42,98.25,123,'China_LowRes')