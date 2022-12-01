import pygrib
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


class atmosphere_analysis:
    def __init__(self, EAC4, EGG4, year, month, day, hour, latitude, longitude):
        self.EAC4 = pygrib.open(EAC4)
        self.EGG4 = pygrib.open(EGG4)
        self.date = datetime.datetime(year,month,day,hour)
        self.date_1 = datetime.datetime(year-1, month,day,hour)
        self.latitude = latitude
        self.longitude = longitude
    
    def get_gasses(self):
        self.get_eac4()
        self.get_egg4()
        return self
    
    def get_eac4(self):
        self.carbon_monoxide = np.zeros((61,241,480))
        self.formaldehyde = np.zeros((61,241,480))
        self.nitric_acid = np.zeros((61,241,480))
        self.nitrogen_dioxide = np.zeros((61,241,480))
        self.ozone = np.zeros((61,241,480))
        self.sulfur_dioxide = np.zeros((61,241,480))
        self.air_temperature = np.zeros((61,241,480))
        self.surface_air_pressure = np.zeros((241,480))
        for set in self.EAC4:
            if set.validDate == self.date:
                match set.cfName:
                    case 'mass_fraction_of_carbon_monoxide_in_air':
                        self.carbon_monoxide[set.level] = set.values * 1e6
                    case 'mass_fraction_of_formaldehyde_in_air':
                        self.formaldehyde[set.level] = set.values * 1e6
                    case 'mass_fraction_of_nitric_acid_in_air':
                        self.nitric_acid[set.level] = set.values * 1e6
                    case 'mass_fraction of_nitrogen_dioxide_in_air':
                        self.nitrogen_dioxide[set.level] = set.values * 1e6
                    case 'mass_fraction_of_ozone_in_air':
                        self.ozone[set.level] = set.values * 1e6
                    case 'mass_fraction_of_sulfur_dioxide_in_air':
                        self.sulfur_dioxide[set.level] = set.values * 1e6
                    case 'air_temperature':
                        self.air_temperature[set.level] = set.values - 273.15
                    case 'surface_air_pressure':
                        self.surface_air_pressure = set.values * 0.01
        return self

    def get_egg4(self):
        self.carbon_dioxide = np.zeros((61,241,480))
        self.methane = np.zeros((61,241,480))
        for set in self.EGG4:
            if set.validDate == self.date:
                match set.cfName:
                    case 'mass_fraction_of_carbon_dioxide_in_air':
                        self.carbon_dioxide[set.level] = set.values * 1e6
                    case 'mass_fraction_of_methane_in_air':
                        self.methane[set.level] = set.values * 1e6
        return self

    def average_layers(self):
        self.carbon_monoxide = np.average(self.carbon_monoxide,axis=0)
        self.formaldehyde = np.average(self.formaldehyde,axis=0)
        self.nitric_acid = np.average(self.nitric_acid,axis=0)
        self.nitrogen_dioxide = np.average(self.nitrogen_dioxide,axis=0)
        self.ozone = np.average(self.ozone,axis=0)
        self.sulfur_dioxide = np.average(self.sulfur_dioxide,axis=0)
        
        self.carbon_dioxide = np.average(self.carbon_dioxide,axis=0)
        self.methane = np.average(self.methane,axis=0)

        self.two_m_temperature = self.air_temperature[-1]
        self.air_temperature = np.average(self.air_temperature,axis=0)
        return self
 
    def get_location(self):
        self.latitudes, self.longitudes = self.EAC4[1].latlons()
        self.EAC4.rewind()
        self.arg_latitude = np.argmin(np.abs(self.latitude - self.latitudes[:,0]))
        self.arg_longitude = np.argmin(np.abs(self.longitude - self.longitudes[:,0]))

        self.carbon_monoxide = self.carbon_monoxide[self.arg_latitude,self.arg_longitude]
        self.formaldehyde = self.formaldehyde[self.arg_latitude,self.arg_longitude]
        self.nitric_acid = self.nitric_acid[self.arg_latitude,self.arg_longitude]
        self.nitrogen_dioxide = self.nitrogen_dioxide[self.arg_latitude,self.arg_longitude]
        self.ozone = self.ozone[self.arg_latitude,self.arg_longitude]
        self.sulfur_dioxide = self.sulfur_dioxide[self.arg_latitude,self.arg_longitude]

        self.two_m_temperature = self.two_m_temperature[self.arg_latitude,self.arg_longitude]
        self.air_temperature = self.air_temperature[self.arg_latitude,self.arg_longitude]

        self.surface_air_pressure = self.surface_air_pressure[self.arg_latitude,self.arg_latitude]

        self.carbon_dioxide = self.carbon_dioxide[self.arg_latitude,self.arg_longitude]
        self.methane = self.methane[self.arg_latitude,self.arg_longitude]

        return self

if __name__ == '__main__':
    A = atmosphere_analysis('EAC4.grib','EGG4.grib',2021,11,29,0,54.767273,-1.568486)
    A.get_gasses()
    A.average_layers()
    A.get_location()

