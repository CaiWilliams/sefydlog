from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from smarts import * 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import smarts
from timezonefinder import TimezoneFinder
import pytz
from GPVDM import *
import multiprocessing
import tqdm
import pandas as pd

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
        data = pd.read_csv('latlons.csv')

        self.latitudes = data['latitudes'].dropna(axis=0).to_numpy()
        self.longitudes = data['longitudes'].to_numpy()
        self.longitudes = (self.longitudes + 180) % 360 - 180
        self.arg_latitude = np.argmin(np.abs(latitude - self.latitudes))
        self.arg_longitude = np.argmin(np.abs(longitude - self.longitudes))
        #gasses = [self.carbon_monoxide, self.sulfur_dioxide, self.nitric_acid, self.water_vapour, self.methane, self.TAU550, self.formaldehyde, self.air_temperature, self.nitrogen_dioxide, self.ozone, self.carbon_dioxide, self.surface_air_pressure]
        self.carbon_monoxide = self.carbon_monoxide[self.arg_latitude,self.arg_longitude]
        self.sulfur_dioxide = self.sulfur_dioxide[self.arg_latitude,self.arg_longitude]
        #self.two_m_temperature = self.two_m_temperature[self.arg_latitude,self.arg_longitude]
        self.nitric_acid = self.nitric_acid[self.arg_latitude,self.arg_longitude]
        self.water_vapour = self.water_vapour[self.arg_latitude,self.arg_longitude]
        if self.water_vapour > 12:
            self.water_vapour = 12
        self.methane = self.methane[self.arg_latitude,self.arg_longitude]
        self.TAU550 = self.TAU550[self.arg_latitude,self.arg_longitude]
        if self.TAU550 > 5:
            self.TAU550 = 4.99
        self.formaldehyde = self.formaldehyde[self.arg_latitude,self.arg_longitude]
        self.air_temperature = self.air_temperature[self.arg_latitude, self.arg_longitude] - 273.15
        self.nitrogen_dioxide = self.nitrogen_dioxide[self.arg_latitude, self.arg_longitude]
        self.ozone = self.ozone[self.arg_latitude,self.arg_longitude]
        self.carbon_dioxide = self.carbon_dioxide[self.arg_latitude,self.arg_longitude]
        self.surface_air_pressure = self.surface_air_pressure[self.arg_latitude,self.arg_longitude]
        self.relative_humidity = self.relative_humidity[self.arg_latitude,self.arg_longitude]
        gasses = [self.carbon_monoxide, self.sulfur_dioxide, self.nitric_acid, self.water_vapour, self.methane, self.TAU550, self.formaldehyde, self.air_temperature, self.nitrogen_dioxide, self.ozone, self.carbon_dioxide, self.surface_air_pressure, self.relative_humidity]
        return gasses


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
    
    def generate_spectrum_standalone(latitude, longitude, date, data):
        tf = TimezoneFinder()
        lat = latitude
        long = longitude
        timezone = tf.timezone_at(lat=lat,lng=long)
        timezone = datetime.datetime.now(pytz.timezone(str(timezone)))
        timezone = timezone.utcoffset().total_seconds()/60/60
        carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, TAU550, formaldehyde, air_temperature, nitrogen_dioxide, oz, carbon_di, surface_air_pressure,relative_humidity = data

        if latitude < 0:
            if date.month <= 3 or date.month >= 9:
                season = 'SUMMER'
            else:
                season = 'WINTER'
        else:
            if date.month <= 3 or date.month >= 9:
                season = 'WINTER'
            else:
                season = 'SUMMER'
        try:
            wavelength,intensity = spectrum(surface_air_pressure/100,0,air_temperature,relative_humidity,season,air_temperature,formaldehyde,methane,carbon_monoxide,nitric_acid,nitrogen_dioxide,oz,sulfur_dioxide,carbon_di,'S&F_RURAL',TAU550,water_vapour,date.year,date.month,date.day,date.hour,latitude,longitude,timezone) 
        except:
            wavelength = 0 
            intensity = 0
        return wavelength, intensity
    
    def generate_spectrum_pristine_standalone(latitude, longitude, date, data):
        tf = TimezoneFinder()
        lat = latitude
        long = longitude
        timezone = tf.timezone_at(lat=lat,lng=long)
        timezone = datetime.datetime.now(pytz.timezone(str(timezone)))
        timezone = timezone.utcoffset().total_seconds()/60/60
        carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, TAU550, formaldehyde, air_temperature, nitrogen_dioxide, oz, carbon_di, surface_air_pressure,relative_humidity = data
        if latitude < 0:
            if date.month <= 3 or date.month >= 9:
                season = 'SUMMER'
            else:
                season = 'WINTER'
        else:
            if date.month <= 3 or date.month >= 9:
                season = 'WINTER'
            else:
                season = 'SUMMER'
        try:
            wavelength,intensity = spectrum_pristine(surface_air_pressure/100,0,air_temperature,relative_humidity,season,air_temperature,'S&F_RURAL',date.year,date.month,date.day,date.hour,latitude,longitude,timezone) 
        except:
            wavelength = 0 
            intensity = 0
        return wavelength, intensity

    
    def generate_spectrum_pristine(self):
        tf = TimezoneFinder()
        lat = self.latitude
        long = self.longitude
        timezone = tf.timezone_at(lat=lat,lng=long)
        timezone = datetime.datetime.now(pytz.timezone(str(timezone)))
        timezone = timezone.utcoffset().total_seconds()/60/60
        carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, TAU550, formaldehyde, air_temperature, nitrogen_dioxide, oz, carbon_di, surface_air_pressure,relative_humidity = data
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
            wavelength,intensity = spectrum_pristine(self.surface_air_pressure/100,0,self.air_temperature,self.relative_humidity,season,self.air_temperature,'S&F_RURAL',self.year,self.month,self.day,self.hour,self.latitude,self.longitude,timezone) 
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
    data = data[4]
    lock.acquire()
    wavelength, intensity = Atmosphere.generate_spectrum_standalone(latitude,longitude,date,data)
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

def save_to_oghma_mp_pristine(data):
    longitude = data[0]
    latitude = data[1]
    idx = data[2]
    date = data[3]
    data = data[4]
    lock.acquire()
    wavelength, intensity = Atmosphere.generate_spectrum_pristine_standalone(latitude,longitude,date,data)
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

def run_oghma_mp(G,idx,temp,name):
    G.create_job_json('TempDevice'+str(idx),name)
    G.modify_pm_json('virtual_spectra','light_spectra',"light_spectrum",category=['optical','light_sources','lights'],layer_name="segment",layer_number=0,value='temp'+str(idx))
    G.modify_pm_json('set_point',category=['thermal'],value=temp)
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
    results = pd.DataFrame(columns=['Date','PCE','Pmax','FF','Voc','Jsc','Air Temperature','Carbon Dioxide','Carbon Monoxide','Formaldehyde','Methane','Nitric Acid','Nitrogen Dioxide','Ozone','Relative Humididity','Sulfur Dioxide','Air Pressure','TAU550','Water Vapour'])
    data_list = [Atmosphere(date.year,date.month,date.day,date.hour).get_location(longitude, latitude) for date in tqdm.tqdm(dates)]
    data = [[longitude,latitude,idx,date,data_list[idx]] for idx,date in enumerate(tqdm.tqdm(dates))]
    pool = multiprocessing.Pool(processes=128)
    for _ in tqdm.tqdm(pool.imap_unordered(save_to_oghma_mp,data), total=len(data)):
      pass

    G = gpvdm()
    for idx,date in enumerate(dates):
     carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, TAU550, formaldehyde, air_temperature, nitrogen_dioxide, oz, carbon_di, surface_air_pressure,relative_humidity = data_list[idx]
     temp = air_temperature + 273.15
     run_oghma_mp(G,idx,temp,'p3htpcbm')
    G.run()


    for idx,date in enumerate(tqdm.tqdm(dates)):
        #results['Date',idx] = date
        PCE = fetch_result_mp(idx,'pce')
        Pmax = fetch_result_mp(idx,'Pmax')
        FF = fetch_result_mp(idx,'ff')
        Voc = fetch_result_mp(idx,'voc')
        Jsc = fetch_result_mp(idx,'jsc')
        #data = data_list[idx]
        carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, TAU550, formaldehyde, air_temperature, nitrogen_dioxide, oz, carbon_di, surface_air_pressure,relative_humidity = data_list[idx]
        results.loc[idx] = [date,PCE,Pmax,FF,Voc,Jsc,air_temperature,carbon_dioxide,carbon_monoxide,formaldehyde,methane,nitric_acid,nitrogen_dioxide,ozone,relative_humidity,sulfur_dioxide,surface_air_pressure,TAU550,water_vapour]
    results.to_csv(os.path.join(os.getcwd(),'Results',name+'_'+str(year)+'.csv'))

def run_dates_mp_pristine(name,year,longitude,latitude):
    start_date =  datetime.datetime(year,1,1,12)
    end_date = datetime.datetime(year,12,31,21)
    dates = pd.date_range(start=start_date, end=end_date,freq='3H').to_pydatetime().tolist()
    results = pd.DataFrame(columns=['Date','PCE','Pmax','FF','Voc','Jsc','Air Temperature','Carbon Dioxide','Carbon Monoxide','Formaldehyde','Methane','Nitric Acid','Nitrogen Dioxide','Ozone','Relative Humididity','Sulfur Dioxide','Air Pressure','TAU550','Water Vapour'])
    data_list = [Atmosphere(date.year,date.month,date.day,date.hour).get_location(longitude, latitude) for date in tqdm.tqdm(dates)]
    data = [[longitude,latitude,idx,date,data_list[idx]] for idx,date in enumerate(tqdm.tqdm(dates))]
    pool = multiprocessing.Pool(processes=128)
    for _ in tqdm.tqdm(pool.imap_unordered(save_to_oghma_mp_pristine,data), total=len(data)):
      pass

    G = gpvdm()
    for idx,date in enumerate(dates):
     carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, TAU550, formaldehyde, air_temperature, nitrogen_dioxide, oz, carbon_di, surface_air_pressure,relative_humidity = data_list[idx]
     temp = air_temperature + 273.15
     run_oghma_mp(G,idx,temp,'p3htpcbm')
    G.run()

    for idx,date in enumerate(tqdm.tqdm(dates)):
        #results['Date',idx] = date
        PCE = fetch_result_mp(idx,'pce')
        Pmax = fetch_result_mp(idx,'Pmax')
        FF = fetch_result_mp(idx,'ff')
        Voc = fetch_result_mp(idx,'voc')
        Jsc = fetch_result_mp(idx,'jsc')
        #data = data_list[idx]
        carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, TAU550, formaldehyde, air_temperature, nitrogen_dioxide, oz, carbon_di, surface_air_pressure,relative_humidity = data_list[idx]
        results.loc[idx] = [date,PCE,Pmax,FF,Voc,Jsc,air_temperature,carbon_dioxide,carbon_monoxide,formaldehyde,methane,nitric_acid,nitrogen_dioxide,ozone,relative_humidity,sulfur_dioxide,surface_air_pressure,TAU550,water_vapour]
    results.to_csv(os.path.join(os.getcwd(),'Results',name+'_'+str(year)+'.csv'))

if __name__ == '__main__':
    global lock
    lock = multiprocessing.Lock()

    #Covid

    run_dates_mp('P3HTPCBM_Mexico_City',2019,-99.129631,19.422804)
    run_dates_mp_pristine('P3HTPCBM_Mexico_City_Pristine',2019,-99.129631,19.422804)

    run_dates_mp('P3HTPCBM_Zhengzhou',2019,113.686424,34.756545)
    run_dates_mp_pristine('P3HTPCBM_Zhengzhou_Pristine',2019,113.686424,34.756545)

    run_dates_mp('P3HTPCBM_Lahore',2019,74.370776,31.519601)
    run_dates_mp_pristine('P3HTPCBM_Lahore_Pristine',2019,74.370776,31.519601)

    run_dates_mp('P3HTPCBM_Durham',2019,-1.5687486,54.767273)
    run_dates_mp_pristine('P3HTPCBM_Durham_Pristine',2019,-1.5687486,54.767273)

    run_dates_mp('P3HTPCBM_San_Juan',2019,-66.123175,18.371388)
    run_dates_mp_pristine('P3HTPCBM_San_Juan_Pristine',2019,-66.123175,18.371388)

    run_dates_mp('P3HTPCBM_Mexico_City',2020,-99.129631,19.422804)
    run_dates_mp_pristine('P3HTPCBM_Mexico_City_Pristine',2020,-99.129631,19.422804)

    run_dates_mp('P3HTPCBM_Zhengzhou',2020,113.686424,34.756545)
    run_dates_mp_pristine('P3HTPCBM_Zhengzhou_Pristine',2020,113.686424,34.756545)

    run_dates_mp('P3HTPCBM_Lahore',2020,74.370776,31.519601)
    run_dates_mp_pristine('P3HTPCBM_Lahore_Pristine',2020,74.370776,31.519601)

    run_dates_mp('P3HTPCBM_Durham',2020,-1.5687486,54.767273)
    run_dates_mp_pristine('P3HTPCBM_Durham_Pristine',2020,-1.5687486,54.767273)

    run_dates_mp('P3HTPCBM_San_Juan',2020,-66.123175,18.371388)
    run_dates_mp_pristine('P3HTPCBM_San_Juan_Pristine',2020,-66.123175,18.371388)

    #Wild Fires

    run_dates_mp('P3HTPCBM_Santa_Clarita',2020,-118.534802,34.387461)
    run_dates_mp_pristine('P3HTPCBM_Santa_Clarita_Pristine',2020,-118.534802,34.387461)

    run_dates_mp('P3HTPCBM_Santa_Clarita',2019,-118.534802,34.387461)
    run_dates_mp_pristine('P3HTPCBM_Santa_Clarita_Pristine',2019,-118.534802,34.387461)

    run_dates_mp('P3HTPCBM_Malibu',2020,-118.785394,34.029256)
    run_dates_mp_pristine('P3HTPCBM_Malibu_Pristine',2020,-118.785394,34.029256)

    run_dates_mp('P3HTPCBM_Malibu',2019,-118.785394,34.029256)
    run_dates_mp_pristine('P3HTPCBM_Malibu_Pristine',2019,-118.785394,34.029256)

    run_dates_mp('P3HTPCBM_Santa_Rosa',2020,-122.772402,38.491299)
    run_dates_mp_pristine('P3HTPCBM_Santa_Rosa_Pristine',2020,-122.772402,38.491299)

    run_dates_mp('P3HTPCBM_Santa_Rosa',2019,-122.772402,38.491299)
    run_dates_mp_pristine('P3HTPCBM_Santa_Rosa_Pristine',2019,-122.772402,38.491299)

    #Volcano

    run_dates_mp('P3HTPCBM_Durham',2010,-1.5687486,54.767273)
    run_dates_mp_pristine('P3HTPCBM_Durham_Pristine',2010,-1.5687486,54.767273)

    run_dates_mp('P3HTPCBM_Montreal',2010,-73.560738,45.493144)
    run_dates_mp_pristine('P3HTPCBM_Montreal_Pristine',2010,-73.560738,45.493144)

    run_dates_mp('P3HTPCBM_Reykjavik',2010,-21.82895,64.117009)
    run_dates_mp_pristine('P3HTPCBM_Reykjavik_Pristine',2010,-21.82895,64.117009)

    #Olympics

    run_dates_mp('P3HTPCBM_Beijing',2008,116.533129,39.9154649)
    run_dates_mp_pristine('P3HTPCBM_Beijing_Pristine',2008,116.533129,39.9154649)

    run_dates_mp('P3HTPCBM_Guangzhou',2008,113.288508,23.106849)
    run_dates_mp_pristine('P3HTPCBM_Guangzhou_Pristine',2008,113.288508,23.106849)

    run_dates_mp('P3HTPCBM_Chengdu',2008,104.168877,30.636373)
    run_dates_mp_pristine('P3HTPCBM_Chengdu_Pristine',2008,104.168877,30.636373)

    run_dates_mp('P3HTPCBM_London',2008,-0.010573,51.542710)
    run_dates_mp_pristine('P3HTPCBM_London_Pristine',2008,-0.010573,51.542710)

    run_dates_mp('P3HTPCBM_Rio',2008,-43.178719,-22.917613)
    run_dates_mp_pristine('P3HTPCBM_Rio_Pristine',2008,43.178719,-22.917613)

    run_dates_mp('P3HTPCBM_Tokyo',2008,139.775831,35.667202)
    run_dates_mp_pristine('P3HTPCBM_Tokyo_Pristine',2008,139.775831,35.667202)