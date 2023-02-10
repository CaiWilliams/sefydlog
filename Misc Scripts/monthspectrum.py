from smarts import * 
from gas_ppmv import *
import numpy as np
import copy
from PIL import Image
import datetime
import os
import matplotlib.image
import matplotlib.pyplot as plt
import cv2
import matplotlib.colors
import pandas as pd
import multiprocessing
import tqdm


def norm(X):
    return (X - np.min(X) / (np.max(X) - np.min(X))) 

def array_to_rgb(X):
        R = X/3
        G = X/3
        B = X/3
        return R,G,B

def RGB(X):
    vfunc = np.vectorize(array_to_rgb)
    R,G,B = vfunc(X)
    return np.array([(i,j,k) for i,j,k in zip(B.ravel(),G.ravel(),R.ravel())]).reshape(241,480,3)

def get_map(data):
    EAC4 = data[0]
    EGG4 = data[1]
    A = atmosphere_analysis(EAC4,EGG4,2019,1,1,0,0,0,'cfgrib')
    date = data[2]
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    
    A.date = datetime.datetime(year,month,day,hour)
    A.get_gasses()
    #A.average_layers()
    A.normalise_gasses()
    A.normalised_gasses_to_rgb()




    names = ['carbon_monoxide','formaldehyde','nitric_acid','nitrogen_dioxide','ozone','sulfur_dioxide','air_temperature','carbon_dioxide','methane','TAU550','water_vapour','surface_air_pressure','relative_humidity','top_irradiance']
    #maxs = [np.max(data['carbon monoxide']),np.max(data['formaldehyde']),np.max(data['nitric acid']),np.max(data['nitrogen dioxide']),np.max(data['ozone']),np.max(data['sulfur dioxide']),np.max(data['air temperature']),np.max(data['carbon dioxide']),np.max(data['methane']),np.max(data['TAU550']),np.max(data['water vapour']),np.max(data['surface air pressure']),np.max(data['relative humidity'])]
    #mins = [np.min(data['carbon monoxide']),np.min(data['formaldehyde']),np.min(data['nitric acid']),np.min(data['nitrogen dioxide']),np.min(data['ozone']),np.min(data['sulfur dioxide']),np.min(data['air temperature']),np.min(data['carbon dioxide']),np.min(data['methane']),np.min(data['TAU550']),np.min(data['water vapour']),np.min(data['surface air pressure']),np.min(data['relative humidity'])]
    maxs = [np.max(A.carbon_monoxide),np.max(A.formaldehyde),np.max(A.nitric_acid),np.max(A.nitrogen_dioxide),np.max(A.ozone),np.max(A.sulfur_dioxide),np.max(A.air_temperature),np.max(A.carbon_dioxide),np.max(A.methane),np.max(A.TAU550),np.max(A.water_vapor),np.max(A.surface_air_pressure),np.max(A.relative_humidity),np.max(A.top_irradiance)]
    mins = [np.min(A.carbon_monoxide),np.min(A.formaldehyde),np.min(A.nitric_acid),np.min(A.nitrogen_dioxide),np.min(A.ozone),np.min(A.sulfur_dioxide),np.min(A.air_temperature),np.min(A.carbon_dioxide),np.min(A.methane),np.min(A.TAU550),np.min(A.water_vapor),np.min(A.surface_air_pressure),np.min(A.relative_humidity),np.min(A.top_irradiance)]
    gasses = [A.carbon_monoxide_rgb, A.formaldehyde_rgb, A.nitric_acid_rgb, A.nitrogen_dioxide_rgb, A.ozone_rgb, A.sulfur_dioxide_rgb, A.air_temperature_rgb, A.carbon_dioxide_rgb, A.methane_rgb, A.TAU550_rgb, A.water_vapor_rgb, A.surface_air_pressure_rgb, A.relative_humidity_rgb, A.top_irradiance_rgb]
    #gasses = [RGB(norm(data['carbon monoxide'])), RGB(norm(data['formaldehyde'])), RGB(norm(data['nitric acid'])), RGB(norm(data['nitrogen dioxide'])), RGB(norm(data['ozone'])), RGB(norm(data['sulfur dioxide'])), RGB(norm(data['air temperature'])), RGB(norm(data['carbon dioxide'])), RGB(norm(data['methane'])), RGB(norm(data['TAU550'])), RGB(norm(data['water vapour'])), RGB(norm(data['surface air pressure'])), RGB(norm(data['relative humidity']))]



    for idx in range(len(gasses)):
        B = gasses[idx]*255
        B = B.astype(np.uint8)
        try:
            os.makedirs(os.path.join(os.getcwd(),str(year),str(month),str(day),str(hour)))
        except:
            C = 1
        name = names[idx] +'_MAX' + str(maxs[idx]) + '_MIN' + str(mins[idx]) + '.png'
        #np.savetxt(os.path.join(os.getcwd(),str(year),str(month),str(day),str(hour),name),B, delimiter=',')
        cv2.imwrite(os.path.join(os.getcwd(),str(year),str(month),str(day),str(hour),name),B)


def run(start_date,end_date):
    dates = pd.date_range(start=start_date, end=end_date,freq='3H').to_pydatetime().tolist()
    data = [('EAC4_'+str(date.year)+'.grib','EGG4_'+str(date.year)+'.grib',date) for date in dates]
    
    pool = multiprocessing.Pool(processes=128)
    for _ in tqdm.tqdm(pool.imap_unordered(get_map,data,chunksize=31), total=len(dates)):
         pass


if __name__ == '__main__':
    years = np.arange(2015,2015)
    start_date =  datetime.datetime(2020,12,30,0)
    end_date = datetime.datetime(2020,12,31,21)
    run(start_date,end_date)