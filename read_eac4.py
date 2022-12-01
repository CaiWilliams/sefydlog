import pygrib
import datetime
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm

grbs = pygrib.open('EGG4.grib')
grbs.rewind()
date_valid = datetime.datetime(2020,11,29,0)


m = Basemap(projection='mill', resolution='i')
t2mall = np.zeros((1,241,480))
for x in np.arange(1,61,1):
    grbs.rewind()
    print(x)
    t2m = []
    for grb in grbs:
        if grb.validDate == date_valid and grb.cfName == 'mass_fraction_of_carbon_dioxide_in_air' and grb.level == x:
            print(grb.cfName)
            t2m.append(grb.values)

    t2m = np.array(t2m)
    if len(t2m) != 0:
        t2mall = t2mall + t2m

lat, lons = grb.latlons()
#x,y = m(lons,lat)
Z = (t2mall[0]/60)*1e6
m.pcolor(lons,lat,Z,cmap='gnuplot',latlon=True)
m.colorbar(label='Concentration (ppmv)')
m.drawcoastlines()
#m.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
#m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,60),labels=[0,0,0,1])
plt.tight_layout()
plt.savefig('Carbon_dioxide.png',dpi=1200)