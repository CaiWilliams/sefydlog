import pygrib
import os

path = os.path.join(os.getcwd(),'adaptor.mars.internal-1673008054.380453-17530-4-48f989fa-40b4-46f8-8eb2-9b209a63347e.grib')
EAC4 = pygrib.open(path)
print(EAC4[1].validDate)
