from smarts import *
from GPVDM import *
from Equivelent_Circuit import *
from generate_pristine_concentrations import gas_concentrations


from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime


from timezonefinder import TimezoneFinder
import pytz

import multiprocessing
import tqdm
import pandas as pd
import copy


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
        #try:
        wavelength,intensity = spectrum(self.surface_air_pressure/100,0,self.air_temperature,self.relative_humidity,season,self.air_temperature,self.formaldehyde,self.methane,self.carbon_monoxide,self.nitric_acid,self.nitrogen_dioxide,self.ozone,self.sulfur_dioxide,self.carbon_dioxide,'S&F_RURAL',self.TAU550,self.water_vapour,self.year,self.month,self.day,self.hour,self.latitude,self.longitude,timezone) 
        #except:
        #    wavelength = 0 
        #    intensity = 0
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

    def generate_spectrum_one_at_a_time(latitude, longitude, date, data):
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
            wavelength,intensity = spectrum_refrence_ozone(surface_air_pressure/100,0,air_temperature,relative_humidity,season,air_temperature,formaldehyde,methane,carbon_monoxide,nitric_acid,nitrogen_dioxide,oz,sulfur_dioxide,carbon_di,'S&F_RURAL',TAU550,water_vapour,date.year,date.month,date.day,date.hour,latitude,longitude,timezone) 
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

def fetch_wavlength_intensity(data):
    longitude = data[0]
    latitude = data[1]
    idx = data[2]
    date = data[3]
    data = data[4]
    #lock.acquire()
    wavelength, intensity = Atmosphere.generate_spectrum_standalone(latitude,longitude,date,data)
    os.system('clear')
    #lock.release()
    try:
        wavelength = wavelength*1e-9
        intensity = intensity*1e-9
    except:
        wavelength = 0
        intensity = 0
    df = pd.DataFrame()
    df['Wavelength'] = wavelength
    df['Intensity'] = intensity
    #header = 'Wavelength\tIntensity'
    dir = os.path.join(os.getcwd(),'Temp','Spectrums','temp'+str(idx)+'.csv')
    df.to_csv(dir)
    #np.savetxt(dir, data, delimiter='\t',header=header)
    return

def fetch_wavlength_intensity_pristine(data):
    longitude = data[0]
    latitude = data[1]
    idx = data[2]
    date = data[3]
    data = data[4]
    #lock.acquire()
    wavelength, intensity = Atmosphere.generate_spectrum_pristine_standalone(latitude,longitude,date,data)
    os.system('clear')
    #lock.release()
    try:
        wavelength = wavelength*1e-9
        intensity = intensity*1e-9
    except:
        wavelength = 0
        intensity = 0
    df = pd.DataFrame()
    df['Wavelength'] = wavelength
    df['Intensity'] = intensity
    #header = 'Wavelength\tIntensity'
    dir = os.path.join(os.getcwd(),'Temp','Spectrums','temp'+str(idx)+'.csv')
    df.to_csv(dir)
    #np.savetxt(dir, data, delimiter='\t',header=header)
    return 

def fetch_wavlength_intensity_one_at_a_time(data):
    longitude = data[0]
    latitude = data[1]
    idx = data[2]
    date = data[3]
    data = data[4]
    #lock.acquire()
    wavelength, intensity = Atmosphere.generate_spectrum_one_at_a_time(latitude,longitude,date,data)
    os.system('clear')
    #lock.release()
    try:
        wavelength = wavelength*1e-9
        intensity = intensity*1e-9
    except:
        wavelength = 0
        intensity = 0
    df = pd.DataFrame()
    df['Wavelength'] = wavelength
    df['Intensity'] = intensity
    #header = 'Wavelength\tIntensity'
    dir = os.path.join(os.getcwd(),'Temp','Spectrums','temp'+str(idx)+'.csv')
    df.to_csv(dir)
    #np.savetxt(dir, data, delimiter='\t',header=header)
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

def task(longitude,latitude,date):
        return Atmosphere(date.year,date.month,date.day,date.hour).get_location(longitude, latitude)

def load_loc(data):
    arg_longitude = data[0]
    arg_latitude = data[1]
    date = data[2]
    dir = os.path.join(os.getcwd(),str(date.year),str(date.month),str(date.day),str(date.hour))
    files = os.listdir(dir)
    mins = [float(file.split('_')[-1][3:-4]) for file in files]
    maxs = [float(file.split('_')[-2][3:]) for file in files]
    files_path = [os.path.join(dir,file) for file in files]
    data = [Atmosphere.load_image(path) for path in files_path]
    data = [d[:,:,0]/255 * 3 for d in data]
    data = [d *(maxs[idx]-mins[idx]) + mins[idx] for idx,d in enumerate(data)]
    names = [file.split('_')[-len(file.split('_')):-2] for file in files]
    names = ['_'.join(name) for name in names]
    for idx in range(len(data)):
            match names[idx]:
                case 'carbon_monoxide':#
                    carbon_monoxide = data[idx][arg_latitude,arg_longitude]
                case 'sulfur_dioxide':#
                    sulfur_dioxide = data[idx][arg_latitude,arg_longitude]
                case 'two_m_temperature':
                    two_m_temperature = data[idx][arg_latitude,arg_longitude]
                case 'nitric_acid':#
                    nitric_acid = data[idx][arg_latitude,arg_longitude]
                case 'water_vapour':#
                    water_vapour = data[idx][arg_latitude,arg_longitude]
                case 'methane':#
                    methane = data[idx][arg_latitude,arg_longitude]
                case 'TAU550':#
                    TAU550 = data[idx][arg_latitude,arg_longitude]
                case 'formaldehyde':#
                    formaldehyde = data[idx][arg_latitude,arg_longitude]
                case 'air_temperature':#
                    air_temperature = data[idx][arg_latitude,arg_longitude]
                case 'nitrogen_dioxide':#
                    nitrogen_dioxide = data[idx][arg_latitude,arg_longitude]
                case 'ozone':#
                    ozone = data[idx][arg_latitude,arg_longitude]
                case 'carbon_dioxide':
                    carbon_dioxide = data[idx][arg_latitude,arg_longitude]
                case 'surface_air_pressure':#
                    surface_air_pressure = data[idx][arg_latitude,arg_longitude]
                case 'relative_humidity':#
                    relative_humidity = data[idx][arg_latitude,arg_longitude]
    gasses = [carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, TAU550, formaldehyde, air_temperature, nitrogen_dioxide, ozone, carbon_dioxide, surface_air_pressure, relative_humidity]
    return gasses


def arg_lat_lon(longitude,latitude):
    data = pd.read_csv(os.path.join(os.getcwd(),'Location Lists','latlons.csv'))
    latitudes = data['latitudes'].dropna(axis=0).to_numpy()
    longitudes = data['longitudes'].to_numpy()
    longitudes = (longitudes + 180) % 360 - 180
    arg_latitude = np.argmin(np.abs(latitude - latitudes))
    arg_longitude = np.argmin(np.abs(longitude - longitudes))
    return arg_longitude, arg_latitude

def write(data):
    idx = data[2]
    date = data[3]
    data_list = data[4]
    carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, TAU550, formaldehyde, air_temperature, nitrogen_dioxide, oz, carbon_di, surface_air_pressure,relative_humidity = data_list
    pce,pmax,ff,voc,jsc = run_equivilent_circuit(idx,air_temperature+273.15)
    return [date,pce,pmax,ff,voc,jsc,air_temperature,carbon_di,carbon_monoxide,formaldehyde,methane,nitric_acid,nitrogen_dioxide,oz,relative_humidity,sulfur_dioxide,surface_air_pressure,TAU550,water_vapour]

def run_dates_mp_SDM_pristine(name,start_year,end_year,longitude,latitude):
    arg_longitude, arg_latitude = arg_lat_lon(longitude,latitude)
    start_date =  datetime.datetime(start_year,1,1,0)
    end_date = datetime.datetime(end_year,12,31,21)
    dates = pd.date_range(start=start_date, end=end_date,freq='3H').to_pydatetime().tolist()
    dates_it = [[arg_longitude,arg_latitude,date] for date in dates]
    #results = pd.DataFrame(columns=['Date','PCE','Pmax','FF','Voc','Jsc','Air Temperature','Carbon Dioxide','Carbon Monoxide','Formaldehyde','Methane','Nitric Acid','Nitrogen Dioxide','Ozone','Relative Humididity','Sulfur Dioxide','Air Pressure','TAU550','Water Vapour'])
    data_list = []
    pool = multiprocessing.Pool(processes=100)
    data_list = pool.map(load_loc,dates_it)

    data = [[longitude,latitude,idx,date,data_list[idx]] for idx,date in enumerate(dates)]

    for _ in tqdm.tqdm(pool.imap_unordered(fetch_wavlength_intensity_pristine,data), total=len(data)):
     pass
    
    resutls_r = []
    results_r = pool.map(write,data)
    results = pd.DataFrame(data=results_r,columns=['Date','PCE','Pmax','FF','Voc','Jsc','Air Temperature','Carbon Dioxide','Carbon Monoxide','Formaldehyde','Methane','Nitric Acid','Nitrogen Dioxide','Ozone','Relative Humididity','Sulfur Dioxide','Air Pressure','TAU550','Water Vapour'])
    results.to_csv(os.path.join(os.getcwd(),'Results',name+'.csv'))
    return

def run_dates_mp_SDM(name,start_year,end_year,longitude,latitude):
    arg_longitude, arg_latitude = arg_lat_lon(longitude,latitude)
    start_date =  datetime.datetime(start_year,1,1,0)
    end_date = datetime.datetime(end_year,12,31,21)
    dates = pd.date_range(start=start_date, end=end_date,freq='3H').to_pydatetime().tolist()
    dates_it = [[arg_longitude,arg_latitude,date] for date in dates]
    #results = pd.DataFrame(columns=['Date','PCE','Pmax','FF','Voc','Jsc','Air Temperature','Carbon Dioxide','Carbon Monoxide','Formaldehyde','Methane','Nitric Acid','Nitrogen Dioxide','Ozone','Relative Humididity','Sulfur Dioxide','Air Pressure','TAU550','Water Vapour'])
    data_list = []
    pool = multiprocessing.Pool(processes=100)
    data_list = pool.map(load_loc,dates_it)

    data = [[longitude,latitude,idx,date,data_list[idx]] for idx,date in enumerate(dates)]

    for _ in tqdm.tqdm(pool.imap_unordered(fetch_wavlength_intensity,data), total=len(data)):
     pass
    
    resutls_r = []
    results_r = pool.map(write,data)
    results = pd.DataFrame(data=results_r,columns=['Date','PCE','Pmax','FF','Voc','Jsc','Air Temperature','Carbon Dioxide','Carbon Monoxide','Formaldehyde','Methane','Nitric Acid','Nitrogen Dioxide','Ozone','Relative Humididity','Sulfur Dioxide','Air Pressure','TAU550','Water Vapour'])
    results.to_csv(os.path.join(os.getcwd(),'Results',name+'.csv'))
    return


def run_dates_mp_SDM_one_at_a_time(name,start_year,end_year,longitude,latitude):
    arg_longitude, arg_latitude = arg_lat_lon(longitude,latitude)
    start_date =  datetime.datetime(start_year,1,1,0)
    end_date = datetime.datetime(end_year,12,31,21)
    dates = pd.date_range(start=start_date, end=end_date,freq='3H').to_pydatetime().tolist()
    dates_it = [[arg_longitude,arg_latitude,date] for date in dates]
    #results = pd.DataFrame(columns=['Date','PCE','Pmax','FF','Voc','Jsc','Air Temperature','Carbon Dioxide','Carbon Monoxide','Formaldehyde','Methane','Nitric Acid','Nitrogen Dioxide','Ozone','Relative Humididity','Sulfur Dioxide','Air Pressure','TAU550','Water Vapour'])
    data_list = []
    pool = multiprocessing.Pool(processes=100)
    data_list = np.asarray(pool.map(load_loc,dates_it))

    press = np.asarray([data_list[i][11]/100 for i in range(0,len(data_list))])
    temps = np.asarray([data_list[i][7] for i in range(0,len(data_list))])
    water = np.asarray([data_list[i][3] for i in range(0,len(data_list))])
    rh = np.asarray([data_list[i][12] for i in range(0,len(data_list))])
    gas = gas_concentrations(press,temps,'pristine').T
    temp_pristine = []
    for i in range(np.shape(data_list)[0]):
        print(i)
        print(data_list[i,12])
        temp_pristine.append([gas[i,2],gas[i,8],gas[i,4],0,gas[i,1],0,gas[i,0],temps[i],gas[i,6],0,280,press[i],rh[i]])
    #temp_pristine = [[gas[i,2],gas[i,8],gas[8,4],data_list[3][i],gas[i,1],0,gas[i,0],data_list[i][7],gas[i,6],0,280,data_list[11][i],data_list[12][i]] for i in range(np.shape(data_list)[0])]
    for i in range(0,len(data_list[0])):

        #gas = gas_concentrations(pres,np.asarray(data_list[:][7]),'pristine')
        #print(np.shape(gas))
        #set_zero = np.arange(len(data_list[0]))
        #set_zero = np.delete(set_zero,[3,7,11,12,i])

        #pristine = np.asarray([gas[2],gas[8],gas[4],0,gas[1],0,gas[0],0,gas[6],0,280,0,0])

        #CH2O,CH4,CO,NH02,NH03,NO,NO2,NO3,SO2
        #carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, TAU550, formaldehyde, air_temperature, nitrogen_dioxide, oz, carbon_di, surface_air_pressure,relative_humidity
        #temp_pristine = [[gas[i][2],gas[i][8],gas[i][4],data_list[i][3],gas[i][1],0,gas[i][0],data_list[i][7],gas[i][6],0,280,data_list[i][11],data_list[i][12]] for i in range(0,len(data_list))]

        #pristine = np.asarray([0.19,1e-3,60e-6,0,1.5,0,0,293,1e-3,0.04,280,10000,50])
        temp_data_list = np.zeros(np.shape(data_list))
        temp_data_list = copy.deepcopy(temp_pristine)
        #print(data_list[:][i])
        temp_data_list[:][i] = data_list[:][i]
        #print(np.shape(temp_data_list[:][0]))
        #plt.plot(temp_data_list[:][0])
        # #print(temp_data_list)
        #if i != 10:
        #    temp_data_list[:,10] = 280

        data = [[longitude,latitude,idx,date,temp_data_list[idx]] for idx,date in enumerate(dates)]
        #print(temp_data_list[0])
        for _ in tqdm.tqdm(pool.imap_unordered(fetch_wavlength_intensity_one_at_a_time,data), total=len(data)):
           pass
        
        n = ['carbon_monoxide', 'sulfur_dioxide', 'nitric_acid', 'water_vapour', 'methane', 'TAU550', 'formaldehyde', 'air_temperature', 'nitrogen_dioxide', 'ozone', 'carbon_dioxide', 'surface_air_pressure','relative_humidity']
        resutls_r = []
        results_r = pool.map(write,data)
        results = pd.DataFrame(data=results_r,columns=['Date','PCE','Pmax','FF','Voc','Jsc','Air Temperature','Carbon Dioxide','Carbon Monoxide','Formaldehyde','Methane','Nitric Acid','Nitrogen Dioxide','Ozone','Relative Humididity','Sulfur Dioxide','Air Pressure','TAU550','Water Vapour'])
        results.to_csv(os.path.join(os.getcwd(),'Results',name+'_'+n[i])+'.csv')
        #print(temp_data_list[0])
    return

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
        results.loc[idx] = [date,PCE,Pmax,FF,Voc,Jsc,air_temperature,carbon_di,carbon_monoxide,formaldehyde,methane,nitric_acid,nitrogen_dioxide,oz,relative_humidity,sulfur_dioxide,surface_air_pressure,TAU550,water_vapour]
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
        carbon_monoxide, sulfur_dioxide, nitric_acid, water_vapour, methane, tau550, formaldehyde, air_temperature, nitrogen_dioxide, oz, carbon_di, surface_air_pressure,relative_humidity = data_list[idx]
        results.loc[idx] = [date,PCE,Pmax,FF,Voc,Jsc,air_temperature,carbon_di,carbon_monoxide,formaldehyde,methane,nitric_acid,nitrogen_dioxide,oz,relative_humidity,sulfur_dioxide,surface_air_pressure,tau550,water_vapour]
    results.to_csv(os.path.join(os.getcwd(),'Results',name+'_'+str(year)+'.csv'))

def arleady_exists(f):
    return os.path.exists(os.path.join(os.getcwd(),'Results',f+'.csv'))

if __name__ == '__main__':
    #global lock
    #lock = multiprocessing.Lock()
    locs = pd.read_csv(os.path.join(os.getcwd(),'Location Lists','BritishIsles.csv'))
    years = np.arange(2003,2021)
    rpbar = tqdm.tqdm(total=len(years)*len(locs)*13,mininterval=0)
    for year in years:
        for i in range(len(locs)):
            name = locs.loc[i]['Name']
            state = locs.loc[i]['State']
            lat = float(locs.loc[i]['Latitude'])
            lon = float(locs.loc[i]['Longitude'])
            f = 'PERC_' + str(name) + '_' + str(state) + '_' + str(year)
            fp = 'PERC_' + str(name) + '_' + str(state) + '_pristine_' + str(year)
            file_status = arleady_exists(f)
            time.sleep(0.01)
            #print(f)
            #print(os.path.join(os.getcwd(),'Results',f+'.csv'))
            #print(file_status)
            #if file_status == False:
            run_dates_mp_SDM_one_at_a_time(f,year,year,lon,lat)
            rpbar.update(1)
                #run_dates_mp_SDM_pristine(fp,year,year,lon,lat)
                #rpbar.update(1)
            #else:
            #    rpbar.update(1)
            #    rpbar.update(1)
        #run_dates_mp_SDM_one_at_a_time('PERC_Caracas_Venezuela_'+str(year),year,year,-66.9036,10.4806)
        # run_dates_mp_SDM('PERC_Maracaibo_Venezuela_'+str(year),year,year,-71.6125,10.6427)
        # run_dates_mp_SDM('PERC_Valencia_Venezuela_'+str(year),year,year,-67.9927,10.1579)
        # run_dates_mp_SDM_pristine('PERC_Caracas_Venezuela_pristine'+str(year),year,year,-66.9036,10.4806)
        # run_dates_mp_SDM_pristine('PERC_Maracaibo_Venezuela_pristine'+str(year),year,year,-71.6125,10.6427)
        # run_dates_mp_SDM_pristine('PERC_Valencia_Venezuela_pristine'+str(year),year,year,-67.9927,10.1579)
        # run_dates_mp_SDM('PERC_Mexico_City_'+str(year),year,year,-99.129631,19.422804)
        # run_dates_mp_SDM('PERC_Zhengzhou_'+str(year),year,year,113.686424,34.756545)
        # run_dates_mp_SDM('PERC_Lahore_'+str(year),year,year,74.370776,31.519601)
        # run_dates_mp_SDM('PERC_Durham_'+str(year),year,year,-1.5687486,54.767273)
        # run_dates_mp_SDM('PERC_San_Juan_'+str(year),year,year,-66.123175,18.371388)

        # run_dates_mp_SDM_pristine('PERC_Mexico_City_pristine_'+str(year),year,year,-99.129631,19.422804)
        # run_dates_mp_SDM_pristine('PERC_Zhengzhou_pristine_'+str(year),year,year,113.686424,34.756545)
        # run_dates_mp_SDM_pristine('PERC_Lahore_pristine_'+str(year),year,year,74.370776,31.519601)
        # run_dates_mp_SDM_pristine('PERC_Durham_pristine_'+str(year),year,year,-1.5687486,54.767273)
        # run_dates_mp_SDM_pristine('PERC_San_Juan_pristine_'+str(year),year,year,-66.123175,18.371388)

        # run_dates_mp_SDM('PERC_Santa_Clarita_'+str(year),year,year,-118.534802,34.387461)
        # run_dates_mp_SDM('PERC_Malibu_'+str(year),year,year,-118.785394,34.029256)
        # run_dates_mp_SDM('PERC_Santa_Rosa_'+str(year),year,year,-122.772402,38.491299)

        # run_dates_mp_SDM_pristine('PERC_Santa_Clarita_pristine_'+str(year),year,year,-118.534802,34.387461)
        # run_dates_mp_SDM_pristine('PERC_Malibu_pristine_'+str(year),year,year,-118.785394,34.029256)
        # run_dates_mp_SDM_pristine('PERC_Santa_Rosa_pristine_'+str(year),year,year,-122.772402,38.491299)

        # run_dates_mp_SDM('PERC_Montreal_'+str(year),year,year,-73.560738,45.493144)
        # run_dates_mp_SDM('PERC_Reykjavik_'+str(year),year,year,-21.82895,64.117009)

        # run_dates_mp_SDM_pristine('PERC_Montreal_pristine_'+str(year),year,year,-73.560738,45.493144)
        # run_dates_mp_SDM_pristine('PERC_Reykjavik_pristine_'+str(year),year,year,-21.82895,64.117009)

        # run_dates_mp_SDM('PERC_Beijing_'+str(year),year,year,116.533129,39.9154649)
        # run_dates_mp_SDM('PERC_Guangzhou_'+str(year),year,year,113.288508,23.106849)
        # run_dates_mp_SDM('PERC_Chengdu_'+str(year),year,year,104.168877,30.636373)
        # run_dates_mp_SDM('PERC_London_'+str(year),year,year,-0.010573,51.542710)
        # run_dates_mp_SDM('PERC_Rio_'+str(year),year,year,-43.178719,-22.917613)
        # run_dates_mp_SDM('PERC_Tokyo_'+str(year),year,year,139.775831,35.667202)

        # run_dates_mp_SDM_pristine('PERC_Beijing_pristine_'+str(year),year,year,116.533129,39.9154649)
        # run_dates_mp_SDM_pristine('PERC_Guangzhou_pristine_'+str(year),year,year,113.288508,23.106849)
        # run_dates_mp_SDM_pristine('PERC_Chengdu_pristine_'+str(year),year,year,104.168877,30.636373)
        # run_dates_mp_SDM_pristine('PERC_London_pristine_'+str(year),year,year,-0.010573,51.542710)
        # run_dates_mp_SDM_pristine('PERC_Rio_pristine_'+str(year),year,year,-43.178719,-22.917613)
        # run_dates_mp_SDM_pristine('PERC_Tokyo_pristine_'+str(year),year,year,139.775831,35.667202)

    # #Olympics

    # run_dates_mp('P3HTPCBM_Beijing',2007,116.533129,39.9154649)        if i != 10:
    #        temp_data_list[:,10] = 280
    # run_dates_mp('P3HTPCBM_London',2007,-0.010573,51.542710)
    # run_dates_mp('P3HTPCBM_London',2008,-0.010573,51.542710)
    # run_dates_mp_pristine('P3HTPCBM_London_Pristine',2007,-0.010573,51.542710)
    # #run_dates_mp_pristine('P3HTPCBM_London_Pristine',2008,-0.010573,51.542710)

    # run_dates_mp('P3HTPCBM_Rio',2007,-43.178719,-22.917613)
    # run_dates_mp('P3HTPCBM_Rio',2008,-43.178719,-22.917613)
    #run_dates_mp_pristine('P3HTPCBM_Rio_Pristine',2007,43.178719,-22.917613)
    # #run_dates_mp_pristine('P3HTPCBM_Rio_Pristine',2008,43.178719,-22.917613)

    # run_dates_mp('P3HTPCBM_Tokyo',2007,139.775831,35.667202)
    # run_dates_mp('P3HTPCBM_Tokyo',2008,139.775831,35.667202)
    # run_dates_mp_pristine('P3HTPCBM_Tokyo_Pristine',2007,139.775831,35.667202)
    # #run_dates_mp_pristine('P3HTPCBM_Tokyo_Pristine',2008,139.775831,35.667202)