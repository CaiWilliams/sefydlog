import pygrib
import os

path = os.path.join(os.getcwd(),'EGG4/adaptor.mars_constrained.external-1674218736.3758214-29378-12-e02a647f-5901-40f9-83ea-c5cd146acddf.grib')
EAC4 = pygrib.open(path)
print(EAC4[1].validDate)
