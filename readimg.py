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
from GPVDM import *
import multiprocessing
import tqdm

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
        self.latitudes, self.longitudes = np.genfromtxt('latlons.csv',delimiter=',')
        print(np.genfromtxt('latlons.csv',delimiter=','))
        self.arg_latitude = np.argmin(np.abs(latitude - self.latitudes))
        self.arg_longitude = np.argmin(np.abs(longitude - self.longitudes))
        print(self.arg_latitude,self.arg_longitude)
        #gasses = [self.carbon_monoxide, self.sulfur_dioxide, self.two_m_temperature, self.nitric_acid, self.water_vapour, self.methane, self.TAU550, self.formaldehyde, self.air_temperature, self.nitrogen_dioxide, self.ozone, self.carbon_dioxide, self.surface_air_pressure]
        self.carbon_monoxide = self.carbon_monoxide[self.arg_latitude,self.arg_longitude]
        self.sulfur_dioxide = self.sulfur_dioxide[self.arg_latitude,self.arg_longitude]
        #self.two_m_temperature = self.two_m_temperature[self.arg_latitude,self.arg_longitude]
        self.nitric_acid = self.nitric_acid[self.arg_latitude,self.arg_longitude]
        self.water_vapour = self.water_vapour[self.arg_latitude,self.arg_longitude]
        if self.water_vapour > 12:
            self.water_vapour = 12
        self.methane = self.methane[self.arg_latitude,self.arg_longitude]
        self.TAU550 = self.TAU550[self.arg_latitude,self.arg_longitude]
        self.formaldehyde = self.formaldehyde[self.arg_latitude,self.arg_longitude]
        self.air_temperature = self.air_temperature[self.arg_latitude, self.arg_longitude] - 273.15
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
        try:
            wavelength,intensity = spectrum(self.surface_air_pressure/100,0,self.air_temperature,self.relative_humidity,season,self.air_temperature,self.formaldehyde,self.methane,self.carbon_monoxide,self.nitric_acid,self.nitrogen_dioxide,self.ozone,self.sulfur_dioxide,self.carbon_dioxide,'S&F_RURAL',self.TAU550,self.water_vapour,self.year,self.month,self.day,self.hour,self.latitude,self.longitude,timezone) 
        except:
            wavelength = 0 
            intensity = 0
        return wavelength, intensity


def plot():
    start_date =  datetime.datetime(2019,1,1,12)
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

def save_to_oghma(longitude, latitude, year, month, day, hour):
    date = datetime.datetime(year, month, day, hour)
    data = Atmosphere(date.year, date.month, date.day,date.hour)
    data = data.get_location(longitude,latitude)
    wavelength, intensity = data.generate_spectrum()
    data = np.array([wavelength*1e-9,intensity*1e9]).T
    file_name = 'spectra.inp'
    #print(os.path.join('home','paul','oghma_local','spectra','temp'))
    try:
        os.makedirs(os.path.join('/home','paul','oghma_local','spectra','temp'))
    except:
        print('')
    dir = os.path.join('/home','paul','oghma_local','spectra','temp','spectra.inp')
    header = 'gpvdm\ntitle Light intensity of AM 1.5 Sun\ntype xy\nx_mul 1.0\ny_mul 1000000000.0\nz_mul 1.0\ndata_mul 1.0\nx_label\ny_label Wavelength\nz_label \ndata_label Intensity\nx_units \ny_units nm\nz_units \ndata_units m^{-1}.W.m(^-2)\nlogy False\nlogx False\nlogz False\ntime 0.0\nVexternal 0.0'
    np.savetxt(dir, data, delimiter='\t',header=header)
    return

def save_to_oghma_mp(data):
    longitude = data[0]
    latitude = data[1]
    idx = data[2]
    date = data[3]
    data = Atmosphere(date.year, date.month, date.day,date.hour)
    data = data.get_location(longitude,latitude)
    lock.acquire()
    wavelength, intensity = data.generate_spectrum()
    lock.release()
    os.system('clear')
    try:
        data = np.array([wavelength*1e-9,intensity*1e9]).T
    except:
        data = np.array([0,0]).T
    file_name = 'spectra.inp'
    #print(os.path.join('home','paul','oghma_local','spectra','temp'))
    try:
        os.makedirs(os.path.join('/home','paul','oghma_local','spectra','temp'+str(idx)))
    except:
        print('')
    src = os.path.join('/home','paul','oghma_local','spectra','AM1.5G','data.json')
    dst = os.path.join('/home','paul','oghma_local','spectra','temp'+str(idx),'data.json')
    shutil.copyfile(src,dst)
    dir = os.path.join('/home','paul','oghma_local','spectra','temp'+str(idx),'spectra.inp')
    header = 'gpvdm\ntitle Light intensity of AM 1.5 Sun\ntype xy\nx_mul 1.0\ny_mul 1000000000.0\nz_mul 1.0\ndata_mul 1.0\nx_label\ny_label Wavelength\nz_label \ndata_label Intensity\nx_units \ny_units nm\nz_units \ndata_units m^{-1}.W.m(^-2)\nlogy False\nlogx False\nlogz False\ntime 0.0\nVexternal 0.0'
    np.savetxt(dir, data, delimiter='\t',header=header)
    return

def run_oghma(idx,name):
    G = gpvdm()
    G.create_job_json('TempDevice',name)
    G.save_job()
    G.run()

def run_oghma_mp(G,idx,name):
    G.create_job_json('TempDevice'+str(idx),name)
    G.modify_pm_json('virtual_spectra','light_spectra',"light_spectrum",category=['optical','light_sources','lights'],layer_name="segment",layer_number=0,value='temp'+str(idx))
    G.save_job()
    return G

def fetch_result(value):
    file = os.path.join(os.getcwd(),'GPVDM','TempDevice','sim_info.dat')
    data = pd.read_json(file,typ='series')
    return data[value]

def fetch_result_mp(idx,value):
    file = os.path.join(os.getcwd(),'GPVDM','TempDevice'+str(idx),'sim_info.dat')
    data = pd.read_json(file,typ='series')
    return data[value]


def run_dates():
    start_date =  datetime.datetime(2019,1,1,12)
    end_date = datetime.datetime(2019,12,31,21)
    dates = pd.date_range(start=start_date, end=end_date,freq='3H').to_pydatetime().tolist()
    results = np.zeros(len(dates))
    for idx,date in enumerate(dates):
        save_to_oghma(-99.129631,19.422804,date.year,date.month,date.day,date.hour)
        run_oghma('p3htpcbm')
        results[idx] = fetch_result('pce')
        print(results[idx])
    plt.plot(dates,results)
    plt.ylabel('PCE (%)')
    plt.show()

def run_dates_mp(name,year,longitude,latitude):
    start_date =  datetime.datetime(year,1,1,12)
    end_date = datetime.datetime(year,12,31,21)
    dates = pd.date_range(start=start_date, end=end_date,freq='3H').to_pydatetime().tolist()
    results = pd.DataFrame(columns=['Date','PCE','FF','Voc','Jsc','Air Temperature','Carbon Dioxide','Carbon Monoxide','Formaldehyde','Methane','Nitric Acid','Nitrogen Dioxide','Ozone','Relative Humididity','Sulfur Dioxide','Air Pressure','TAU550','Water Vapour'])
    data = [[longitude,latitude,idx,date]for idx,date in enumerate(dates)]
    pool = multiprocessing.Pool(processes=128)
    #for _ in tqdm.tqdm(pool.imap_unordered(save_to_oghma_mp,data), total=len(data)):
    #     pass

    #G = gpvdm()
    #for idx,date in enumerate(dates):
    #    run_oghma_mp(G,idx,'p3htpcbm')
    #G.run()

    for idx,date in enumerate(tqdm.tqdm(dates)):
        #results['Date',idx] = date
        #PCE = fetch_result_mp(idx,'pce')
        #FF = fetch_result_mp(idx,'ff')
        #Voc = fetch_result_mp(idx,'voc')
        #Jsc = fetch_result_mp(idx,'jsc')
        data = Atmosphere(date.year,date.month,date.day,date.hour)
        data = data.get_location(longitude,latitude)
        #results.loc[idx] = [date,PCE,FF,Voc,Jsc,data.air_temperature,data.carbon_dioxide,data.carbon_monoxide,data.formaldehyde,data.methane,data.nitric_acid,data.nitrogen_dioxide,data.ozone,data.relative_humidity,data.sulfur_dioxide,data.surface_air_pressure,data.TAU550,data.water_vapour]
    #results.to_csv(os.path.join(os.getcwd(),'Results',name+'_'+str(year)+'.csv'))

if __name__ == '__main__':
    global lock
    lock = multiprocessing.Lock()
    run_dates_mp('P3HTPCBM_Mexico_City',2019,-99.129631,19.422804)
    run_dates_mp('P3HTPCBM_Zhengzhou',2019,113.686424,34.756545)
    run_dates_mp('P3HTPCBM_Lahore',2019,74.370776,31.519601)
    run_dates_mp('P3HTPCBM_Durham',2019,-1.5687486,54.767273)
    run_dates_mp('P3HTPCBM_San_Juan',2019,-66.123175,18.371388)
