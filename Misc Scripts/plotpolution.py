from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from smarts import * 


class Atmosphere():
    def __init__(self,year,month,day,hour):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.load_day()
        for idx in range(len(self.data)):
            match self.names[idx]:
                case 'carbon_monoxide':
                    self.carbon_monoxide = self.data[idx]
                case 'sulfur_dioxide':
                    self.sulfur_dioxide = self.data[idx]
                case 'two_m_temperature':
                    self.two_m_temperature = self.data[idx]
                case 'nitric_acid':
                    self.nitric_acid = self.data[idx]
                case 'water_vapour':
                    self.water_vapour = self.data[idx]
                case 'methane':
                    self.methane = self.data[idx]
                case 'TAU550':
                    self.TAU550 = self.data[idx]
                case 'formaldehyde':
                    self.formaldehyde = self.data[idx]
                case 'air_temperature':
                    self.air_temperature = self.data[idx]
                case 'nitrogen_dioxide':
                    self.nitrogen_dioxide = self.data[idx]
                case 'ozone':
                    self.ozone = self.data[idx]
                case 'carbon_dioxide':
                    self.carbon_dioxide = self.data[idx]
                case 'surface_air_pressure':
                    self.surface_air_pressure = self.data[idx]


    def load_image(dir):
        image = Image.open(dir)
        data = np.asarray(image)
        return data

    def get_location(self,longitude,latitude):
        self.latitudes, self.longitudes = np.genfromtxt('latlons.csv')
        self.arg_latitude = np.argmin(np.abs(latitude - self.latitudes))
        self.arg_longitude = np.argmin(np.abs(longitude - self.longitudes))
        #gasses = [self.carbon_monoxide, self.sulfur_dioxide, self.two_m_temperature, self.nitric_acid, self.water_vapour, self.methane, self.TAU550, self.formaldehyde, self.air_temperature, self.nitrogen_dioxide, self.ozone, self.carbon_dioxide, self.surface_air_pressure]
        self.carbon_monoxide = self.carbon_monoxide[self.arg_latitude,self.arg_longitude]
        self.sulfur_dioxide = self.sulfur_dioxide[self.arg_latitude,self.arg_longitude]
        #self.two_m_temperature = self.two_m_temperature[self.arg_latitude,self.arg_longitude]
        self.nitric_acid = self.nitric_acid[self.arg_latitude,self.arg_longitude]
        self.water_vapour = self.water_vapour[self.arg_latitude,self.arg_longitude]
        self.methane = self.methane[self.arg_latitude,self.arg_longitude]
        self.TAU550 = self.TAU550[self.arg_latitude,self.arg_longitude]
        self.formaldehyde = self.formaldehyde[self.arg_latitude,self.arg_longitude]
        self.air_temperature = self.air_temperature[self.arg_latitude, self.arg_longitude]
        self.nitrogen_dioxide = self.nitrogen_dioxide[self.arg_latitude, self.arg_longitude]
        self.ozone = self.ozone[self.arg_latitude,self.arg_longitude]
        self.carbon_dioxide = self.carbon_dioxide[self.arg_latitude,self.arg_longitude]
        self.surface_air_pressure = self.surface_air_pressure[self.arg_latitude,self.arg_longitude]
        return self


    def load_day(self):
        dir = os.path.join(os.getcwd(),str(self.year),str(self.month),str(self.day),str(self.hour))
        files = os.listdir(dir)
        mins = [float(file.split('_')[-1][3:-4]) for file in files]
        maxs = [float(file.split('_')[-2][3:]) for file in files]
        files_path = [os.path.join(dir,file) for file in files]
        data = [Atmosphere.load_image(path) for path in files_path]
        data = [d[:,:,0]/255 * 3 for d in data]
        self.data = [d *(maxs[idx]-mins[idx]) + mins[idx] for idx,d in enumerate(data)]
        names = [file.split('_')[-len(file.split('_')):-2] for file in files]
        self.names = ['_'.join(name) for name in names]
        return self


start_date =  datetime.datetime(2019,1,1,0)
end_date = datetime.datetime(2019,12,31,21)
delta = datetime.timedelta(hours=3)
lat = 54.767273
long = -1.568486
#data = Atmosphere(start_date.year,start_date.month,start_date.day,start_date.hour)
#data = data.get_location(lat,long)
#wavelenth_0, irradiance_0 = spectrum(data.surface_air_pressure,0,data.two_m_temperature,20,'SUMMER',data.air_temperature,0,0,0,0,0,0,0,0,'S&F_URBAN',0,start_date.year,start_date.month,start_date.day,12,float(lat),float(long),-6)
gas = []

while start_date <= end_date:

    data = Atmosphere(start_date.year,start_date.month,start_date.day,start_date.hour)
    data = data.get_location(lat,long)
    gas.append(data.s)
    #plt.imshow(data.carbon_dioxide)
    #plt.colorbar()
    #plt.title(str(start_date.year)+' '+str(start_date.month)+' '+str(start_date.day)+' '+str(start_date.hour))
    #data = data.get_location(lat,long)
    #wavelenth, irradiance = spectrum(data.surface_air_pressure,0,data.air_temperature,20,'SUMMER',data.air_temperature,data.formaldehyde,data.methane,data.carbon_monoxide,data.nitric_acid,data.nitrogen_dioxide,data.ozone,data.sulfur_dioxide,data.carbon_dioxide,'S&F_URBAN',data.TAU550,start_date.year,start_date.month,start_date.day,start_date.hour,float(lat),float(long),-6)
    plt.plot(gas)
    #plt.ylim(bottom=0,top=2.5)
    #plt.plot(wavelenth,irradiance-irradiance_0)
    #plt.ylim(bottom=-0.5,top=0.5)
    #print(data.two_m_temperature)
    plt.draw()
    plt.pause(0.001)
    #plt.savefig(os.path.join(os.getcwd(),'MexicoCity_2010_spectra',str(start_date.day)))
    plt.clf()
    start_date += delta
