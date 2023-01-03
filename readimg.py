from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from smarts import * 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import smarts
from timezonefinder import TimezoneFinder
import pytz

class Atmosphere():
    def __init__(self,year,month,day,hour):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.load_day()
        for idx in range(len(self.data)):
            match self.names[idx]:
                case 'carbon_monoxide':#
                    self.carbon_monoxide = self.data[idx]
                case 'sulfur_dioxide':#
                    self.sulfur_dioxide = self.data[idx]
                case 'two_m_temperature':
                    self.two_m_temperature = self.data[idx]
                case 'nitric_acid':#
                    self.nitric_acid = self.data[idx]
                case 'water_vapour':#
                    self.water_vapour = self.data[idx]
                case 'methane':#
                    self.methane = self.data[idx]
                case 'TAU550':#
                    self.TAU550 = self.data[idx]
                case 'formaldehyde':#
                    self.formaldehyde = self.data[idx]
                case 'air_temperature':#
                    self.air_temperature = self.data[idx]
                case 'nitrogen_dioxide':#
                    self.nitrogen_dioxide = self.data[idx]
                case 'ozone':#
                    self.ozone = self.data[idx]
                case 'carbon_dioxide':
                    self.carbon_dioxide = self.data[idx]
                case 'surface_air_pressure':#
                    self.surface_air_pressure = self.data[idx]
                case 'relative_humidity':#
                    self.relative_humidity = self.data[idx]


    def load_image(dir):
        image = Image.open(dir)
        data = np.asarray(image)
        return data

    def get_location(self,longitude,latitude):
        self.longitude = longitude
        self.latitude = latitude
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
        self.relative_humidity = self.relative_humidity[self.arg_latitude,self.arg_longitude]
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

    def generate_spectrum(self):
        tf = TimezoneFinder()
        lat = self.latitude
        long = self.longitude
        timezone = tf.timezone_at(lat=lat,lng=long)
        timezone = datetime.datetime.now(pytz.timezone(str(timezone)))
        timezone = timezone.utcoffset().total_seconds()/60/60
        if self.latitude < 0:
            if self.month <= 3 or self.month >= 9:
                season = 'SUMMER'
            else:
                season = 'WINTER'
        else:
            if self.month <= 3 or self.month >= 9:
                season = 'WINTER'
            else:
                season = 'SUMMER'
        wavelength,intensity = spectrum(self.surface_air_pressure/100,0,self.air_temperature,self.relative_humidity,season,self.air_temperature,self.formaldehyde,self.methane,self.carbon_monoxide,self.nitric_acid,self.nitrogen_dioxide,self.ozone,self.sulfur_dioxide,self.carbon_dioxide,'S&F_RURAL',self.TAU550,self.water_vapour,self.year,self.month,self.day,self.hour,self.latitude,self.longitude,timezone) 
        return wavelength, intensity

start_date =  datetime.datetime(2003,1,1,12)
end_date = datetime.datetime(2019,12,31,21)
delta = datetime.timedelta(hours=3)
lat = 19.422804
long = -99.129631
gas = []
dates = pd.date_range(start=start_date, end=end_date,freq='3H').to_pydatetime().tolist()
for date in dates:
    data = Atmosphere(date.year,date.month,date.day,date.hour)
    data = data.get_location(long,lat)
    try:
        wavelenthg, intensity = data.generate_spectrum()
    except:
        pass
    #gas.append(data.surface_air_pressure)
    #plt.plot(gas)
    plt.title(str(date.year)+' '+str(date.month)+' '+str(date.day)+' '+str(date.hour))
    plt.plot(wavelenthg,intensity)
    plt.ylim(top=2,bottom=0)
    plt.draw()
    plt.pause(0.001)
    plt.clf()
