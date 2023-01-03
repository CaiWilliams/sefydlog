import pygrib
import os

path = os.path.join(os.getcwd(),'adaptor.mars.internal-1670852522.2302876-13795-11-b32fdbeb-e174-42c7-a310-3afa8a6b5c0d.grib')
EAC4 = pygrib.open(path)
print(EAC4[1].validDate)
