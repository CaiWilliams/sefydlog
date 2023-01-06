import pygrib
import datetime
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from mpl_toolkits.basemap import Basemap
import xarray as xr


class atmosphere_analysis:

    def __init__(self, EAC4, EGG4, year, month, day, hour, latitude, longitude, engine='pygrib'):
        self.year = year
        self.month = month
        self.day = day 
        self.hour = hour
        self.date = datetime.datetime(year,month,day,hour)
        self.date_no_hour = datetime.datetime(year,month,day,0,0,0)
        self.date_string = self.date.strftime('%Y%m%d')
        self.date_1 = datetime.datetime(year-1, month,day,hour)
        self.step = datetime.timedelta(hours=hour)
        self.latitude = latitude
        self.longitude = longitude
        self.engine = engine
        match self.engine:
            case 'pygrib':
                self.EAC4 = pygrib.open(EAC4)
                self.EGG4 = pygrib.open(EGG4)
            case 'cfgrib':
                self.EAC4 = xr.open_dataset(EAC4,engine='cfgrib')
                self.EGG4 = xr.open_dataset(EGG4,engine='cfgrib')
    
    
    def get_gasses(self):
        match self.engine:
            case 'pygrib':
                self.get_eac4()
                self.get_egg4()
                self.values_to_ppmv()
            case 'cfgrib':
                self.get_eac4_cfgrib()
                self.get_egg4_cfgrib()
                self.values_to_ppmv()
        return self
    
    def get_eac4_cfgrib(self):
        self.carbon_monoxide = np.zeros((241,480))
        self.formaldehyde = np.zeros((241,480))
        self.nitric_acid = np.zeros((241,480))
        self.nitrogen_dioxide = np.zeros((241,480))
        self.ozone = np.zeros((241,480))
        self.sulfur_dioxide = np.zeros((241,480))
        self.air_temperature = np.zeros((241,480))
        self.surface_air_pressure = np.zeros((241,480))
        self.water_vapor = np.zeros((241,480))
        self.TAU550 = np.zeros((241,480))
        data = self.EAC4.sel(time=self.date)
        print(data)
        self.carbon_monoxide = data['co'].to_numpy()
        self.formaldehyde = data['hcho'].to_numpy()
        self.nitric_acid = data['hno3'].to_numpy()
        self.nitrogen_dioxide = data['no2'].to_numpy()
        self.ozone = data['go3'].to_numpy()
        self.sulfur_dioxide = data['so2'].to_numpy()
        self.air_temperature_k = data['t'].to_numpy()
        self.air_temperature = data['t'].to_numpy() #- 273.15
        self.surface_air_pressure = data['sp'].to_numpy()
        self.TAU550 = data['aod550'].to_numpy()
        self.water_vapor = data['tcwv'].to_numpy()

    
    def get_egg4_cfgrib(self):
        self.carbon_dioxide = np.zeros((241,480))
        self.methane = np.zeros((241,480))
        self.relative_humidity = np.zeros((241,480))
        self.top_irradiance = np.zeros((241,480))
        data = self.EGG4.sel(time=self.date, step=self.step, method='nearest')
        self.carbon_dioxide = data['co2'].to_numpy()
        self.methane = data['ch4'].to_numpy()
        self.relative_humidity = data['r'].to_numpy()
        self.top_irradiance = data['tisr'].to_numpy()




    def get_eac4(self):
        self.EAC4.rewind()
        self.carbon_monoxide = np.zeros((241,480))
        self.formaldehyde = np.zeros((241,480))
        self.nitric_acid = np.zeros((241,480))
        self.nitrogen_dioxide = np.zeros((241,480))
        self.ozone = np.zeros((241,480))
        self.sulfur_dioxide = np.zeros((241,480))
        self.air_temperature = np.zeros((241,480))
        self.surface_air_pressure = np.zeros((241,480))
        self.water_vapor = np.zeros((241,480))
        self.TAU550 = np.zeros((241,480))
        for set in self.EAC4:
            if set.validDate == self.date and set.level == 1000 or set.validDate == self.date and set.level == 0:
                match set.cfName:
                    case 'mass_fraction_of_carbon_monoxide_in_air':
                        self.carbon_monoxide = set.values
                    case 'mass_fraction_of_formaldehyde_in_air':
                        self.formaldehyde = set.values
                    case 'mass_fraction_of_nitric_acid_in_air':
                        self.nitric_acid = set.values
                    case 'mass_fraction_of_nitrogen_dioxide_in_air':
                        self.nitrogen_dioxide = set.values
                    case 'mass_fraction_of_ozone_in_air':
                        self.ozone = set.values
                    case 'mass_fraction_of_sulfur_dioxide_in_air':
                        self.sulfur_dioxide = set.values
                    case 'air_temperature':
                        self.air_temperature_k = set.values
                        self.air_temperature = set.values #- 273.15
                    case 'surface_air_pressure':
                        self.surface_air_pressure = set.values * 0.01
                    case 'lwe_thickness_of_atmosphere_mass_content_of_water_vapor':
                        self.water_vapor = set.values/1000
                    case 'unknown':
                        self.TAU550 = set.values
        return self
    
    def get_egg4(self):
        self.EGG4.rewind()
        self.carbon_dioxide = np.zeros((241,480))
        self.methane = np.zeros((241,480))
        self.relative_humidity = np.zeros((241,480))

        for set in self.EGG4:
            if set.validDate == self.date_no_hour and set.startStep == self.hour:
                match set.cfName:
                    case 'mass_fraction_of_carbon_dioxide_in_air':
                        self.carbon_dioxide = set.values
                    case 'mass_fraction_of_methane_in_air':
                        self.methane = set.values
                    case 'relative_humidity':
                        self.relative_humidity = set.values
        return self

    def values_to_ppmv(self):
        self.carbon_monoxide = self.carbon_monoxide * 1.22 * 1e6
        self.formaldehyde = self.formaldehyde * 1.22 * 1e6
        self.nitric_acid = self.nitric_acid * 1.22 * 1e6
        self.nitrogen_dioxide = self.nitrogen_dioxide * 1.22 * 1e6 
        self.ozone = self.ozone * 1.22 * 1e6
        self.sulfur_dioxide = self.sulfur_dioxide * 1.22 * 1e6 
        self.carbon_dioxide = self.carbon_dioxide * 1.22 * 1e6
        self.methane = self.methane * 1.22 * 1e6


        universal_gas_constant = 0.082057338
        carbon_monoxide_molmass = 28.01
        formaldehyde_molmass = 30.026
        nitric_acid_molmass = 63.01
        nitrogen_dioxide_molmass = 46.0055
        ozone_molmass = 48
        sulfur_dioxide_molmass = 64.066
        carbon_dioxide_molmass = 44.01
        methane_molmass = 16.04

        self.carbon_monoxide = self.carbon_monoxide * ((universal_gas_constant * self.air_temperature_k)/carbon_monoxide_molmass)
        self.formaldehyde = self.formaldehyde * ((universal_gas_constant * self.air_temperature_k)/formaldehyde_molmass)
        self.nitric_acid = self.nitric_acid * ((universal_gas_constant * self.air_temperature_k)/nitric_acid_molmass)
        self.nitrogen_dioxide = self.nitrogen_dioxide * ((universal_gas_constant * self.air_temperature_k)/nitrogen_dioxide_molmass)
        self.ozone = self.ozone * ((universal_gas_constant * self.air_temperature_k)/ozone_molmass)
        self.sulfur_dioxide = self.sulfur_dioxide * ((universal_gas_constant * self.air_temperature_k)/sulfur_dioxide_molmass)

        self.carbon_dioxide = self.carbon_dioxide * ((universal_gas_constant * self.air_temperature_k)/carbon_dioxide_molmass)
        self.methane = self.methane * ((universal_gas_constant * self.air_temperature_k)/methane_molmass)

        return self

    def to_dict(self):
        data = {'date': self.date}
        data['carbon monoxide'] = self.carbon_dioxide
        data['formaldehyde'] = self.formaldehyde
        data['nitric acid'] = self.nitric_acid
        data['nitrogen dioxide'] = self.nitrogen_dioxide
        data['ozone'] = self.ozone
        data['sulfur dioxide'] = self.sulfur_dioxide
        data['carbon dioxide'] = self.carbon_dioxide
        data['methane'] = self.methane
        data['air temperature'] = self.air_temperature
        data['relative humidity'] = self.relative_humidity
        data['TAU550'] = self.TAU550
        data['water vapour'] = self.water_vapor
        data['surface air pressure'] = self.surface_air_pressure
        return data

    

    def average_layers(self):
        self.carbon_monoxide = np.sum(self.carbon_monoxide,axis=0)/30
        self.formaldehyde = np.sum(self.formaldehyde,axis=0)/30
        self.nitric_acid = np.sum(self.nitric_acid,axis=0)/30
        self.nitrogen_dioxide = np.sum(self.nitrogen_dioxide,axis=0)/30
        self.ozone = np.sum(self.ozone,axis=0)/30
        self.sulfur_dioxide = np.sum(self.sulfur_dioxide,axis=0)/30
        
        self.carbon_dioxide = np.sum(self.carbon_dioxide,axis=0)/30
        self.methane = np.sum(self.methane,axis=0)/30

        self.two_m_temperature = self.air_temperature[-1]
        self.air_temperature = np.sum(self.air_temperature,axis=0)/30
        return self
    
    def get_arg_location(self):
        self.latitudes, self.longitudes = self.EAC4[1].latlons()
        np.savetxt('latlons.csv',[self.latitudes[:,0],self.longitudes[:,0]],delimiter=',')
        self.EAC4.rewind()
        self.arg_latitude = np.argmin(np.abs(self.latitude - self.latitudes[:,0]))
        self.arg_longitude = np.argmin(np.abs(self.longitude - self.longitudes[:,0]))

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

        self.TAU550 = self.TAU550[self.arg_latitude,self.arg_longitude]
        self.water_vapor = self.water_vapor[self.arg_latitude,self.arg_longitude]

        return self
    
    def normalise_gasses(self):
        self.carbon_monoxide_norm = ((self.carbon_monoxide - np.min(self.carbon_monoxide)) / (np.max(self.carbon_monoxide) - np.min(self.carbon_monoxide))) 
        print(np.shape(self.carbon_monoxide_norm))
        self.formaldehyde_norm = (self.formaldehyde - np.min(self.formaldehyde)) / (np.max(self.formaldehyde) - np.min(self.formaldehyde))
        self.nitric_acid_norm = (self.nitric_acid - np.min(self.nitric_acid)) / (np.max(self.nitric_acid) - np.min(self.nitric_acid))
        self.nitrogen_dioxide_norm = (self.nitrogen_dioxide - np.min(self.nitrogen_dioxide)) / (np.max(self.nitrogen_dioxide) - np.min(self.nitrogen_dioxide))
        self.ozone_norm = (self.ozone - np.min(self.ozone)) / (np.max(self.ozone) - np.min(self.ozone))
        self.sulfur_dioxide_norm = (self.sulfur_dioxide - np.min(self.sulfur_dioxide)) / (np.max(self.sulfur_dioxide) - np.min(self.sulfur_dioxide))
        #self.two_m_temperature_norm = (self.two_m_temperature - np.min(self.two_m_temperature)) / (np.max(self.two_m_temperature) - np.min(self.two_m_temperature)) 
        self.air_temperature_norm = (self.air_temperature - np.min(self.air_temperature)) / (np.max(self.air_temperature) - np.min(self.air_temperature))
        self.carbon_dioxide_norm = (self.carbon_dioxide - np.min(self.carbon_dioxide)) / (np.max(self.carbon_dioxide) - np.min(self.carbon_dioxide))
        self.methane_norm = (self.methane - np.min(self.methane)) / (np.max(self.methane) - np.min(self.methane))
        self.TAU550_norm = (self.TAU550 - np.min(self.TAU550)) / (np.max(self.TAU550) - np.min(self.TAU550))
        self.water_vapor_norm = (self.water_vapor - np.min(self.water_vapor)) / (np.max(self.water_vapor) - np.min(self.water_vapor))
        self.surface_air_pressure_norm = (self.surface_air_pressure - np.min(self.surface_air_pressure)) / (np.max(self.surface_air_pressure) - np.min(self.surface_air_pressure))
        self.relative_humidity_norm = (self.relative_humidity - np.min(self.relative_humidity)) / (np.max(self.relative_humidity) - np.min(self.relative_humidity))
        self.top_irradiance_norm = (self.top_irradiance - np.min(self.top_irradiance)) /(np.max(self.top_irradiance) - np.min(self.top_irradiance))
        return self

    def array_to_rgb(X):
        R = X/3
        G = X/3
        B = X/3
        return R,G,B

    def RGB(X):
        vfunc = np.vectorize(atmosphere_analysis.array_to_rgb)
        R,G,B = vfunc(X)
        return np.array([(i,j,k) for i,j,k in zip(B.ravel(),G.ravel(),R.ravel())]).reshape(241,480,3)
        

    def normalised_gasses_to_rgb(self):
        self.carbon_monoxide_rgb = atmosphere_analysis.RGB(self.carbon_monoxide_norm)
        self.formaldehyde_rgb = atmosphere_analysis.RGB(self.formaldehyde_norm)
        self.nitric_acid_rgb = atmosphere_analysis.RGB(self.nitric_acid_norm)
        self.nitrogen_dioxide_rgb = atmosphere_analysis.RGB(self.nitrogen_dioxide_norm)
        self.ozone_rgb = atmosphere_analysis.RGB(self.ozone_norm)
        self.sulfur_dioxide_rgb = atmosphere_analysis.RGB(self.sulfur_dioxide_norm)
        #self.two_m_temperature_rgb = atmosphere_analysis.RGB(self.two_m_temperature_norm)
        self.air_temperature_rgb = atmosphere_analysis.RGB(self.air_temperature_norm)
        self.carbon_dioxide_rgb = atmosphere_analysis.RGB(self.carbon_dioxide_norm)
        self.methane_rgb = atmosphere_analysis.RGB(self.methane_norm)
        self.TAU550_rgb = atmosphere_analysis.RGB(self.TAU550_norm)
        self.water_vapor_rgb = atmosphere_analysis.RGB(self.water_vapor_norm)
        self.surface_air_pressure_rgb = atmosphere_analysis.RGB(self.surface_air_pressure_norm)
        self.relative_humidity_rgb = atmosphere_analysis.RGB(self.relative_humidity_norm)
        self.top_irradiance_rgb = atmosphere_analysis.RGB(self.top_irradiance_norm)

        return self

        

if __name__ == '__main__':
    A = atmosphere_analysis('EAC4_GroundLevel_2019.grib','EGG4_2019_SeaLevel.grib',2019,1,1,0,54.767273,-1.568486,'cfgrib')
    #A.get_arg_location()
    #A.get_gasses()
    #plt.imshow(A.TAU550)
    #plt.colorbar()
    #plt.show()
    #A.average_layers()
    #A.get_location()

