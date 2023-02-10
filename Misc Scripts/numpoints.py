import pandas as pd
import numpy as np
import itertools  
import secrets     

data = pd.read_csv('latlons.csv')
latitudes = data['latitudes'].dropna(axis=0).to_numpy()
longitudes = data['longitudes'].to_numpy()
longitudes = (longitudes + 180) % 360 - 180

num_lat = np.where((latitudes > 46.75) & (latitudes <= 61),latitudes,np.nan)
num_lat = num_lat[~np.isnan(num_lat)]

num_lon = np.where((longitudes > -11) & (longitudes  <= 3.5),longitudes,np.nan)
num_lon = num_lon[~np.isnan(num_lon)]

print(len(num_lat)*len(num_lon))

lat_lons = np.asarray(list(itertools.product(num_lat,num_lon)))
lats = lat_lons[:,0][::2]
lons = lat_lons[:,1][::2]
print(lats)
print(lons)
names = [str(secrets.token_hex(8)) for x in range(len(lats))]

data = pd.DataFrame(index=None)
data['Name'] = names
data['State'] = 'BritishIsles'
data['Latitude'] = lats
data['Longitude'] = lons

#data.to_csv('BritishIsles.csv', index=False)