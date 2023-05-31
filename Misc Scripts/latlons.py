import xarray as xr
import pandas as pd
import numpy as np

ds = xr.open_dataset('EAC4_2019_SeaLevel.grib',engine='cfgrib')
latitudes = ds['latitude'].to_numpy()
longitudes = ds['longitude'].to_numpy()

data = pd.DataFrame()
data['longitudes'] = longitudes
diff = len(longitudes) - len(latitudes)
n = np.empty(diff)
n[:] = np.nan
latitudes = np.append(latitudes,n)
data['latitudes'] = latitudes
print(data)
#data.to_csv('latlons.csv')