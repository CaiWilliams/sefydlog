import pygrib
import os

path = os.path.join(os.getcwd(),'adaptor.mars.internal-1674213501.6338415-17563-8-3904ea5b-e9d0-4503-a6cc-5c0638c68df5.grib')
EAC4 = pygrib.open(path)
print(EAC4[1].validDate)
