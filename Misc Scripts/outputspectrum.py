from smarts import * 
from gas_ppmv import *
import numpy as np
import copy
import PIL



def plot(lats,lons,alt,t,l):
    A = atmosphere_analysis('EAC4_2019.grib','EGG4_2019.grib',2019,6,15,12,lats[0],lons[0])
    A.get_gasses()
    A.average_layers()
    for i in range(len(lats)):
        A.latitude = lats[i]
        A.longitude = lons[i]
        B = copy.copy(A)
        A.get_location()

        wavelenth, irradiance = spectrum(A.surface_air_pressure,alt[i],A.two_m_temperature,30,'SUMMER',A.air_temperature,A.formaldehyde,A.methane,A.carbon_monoxide,A.nitric_acid,A.nitrogen_dioxide,A.ozone,A.sulfur_dioxide,A.carbon_dioxide,'S&F_RURAL',A.TAU550,2019,6,15,12,lats[i],lons[i],t[i])
        data = np.array([wavelenth*1e-9,irradiance*1e9]).T
        plt.plot(wavelenth,irradiance,label=l[i])
        A = B


#dir='spectra.inp'
# wavelenth, irradiance = spectrum(A.surface_air_pressure,0,A.two_m_temperature,30,'SUMMER',A.air_temperature,A.formaldehyde,A.methane,A.carbon_monoxide,A.nitric_acid,A.nitrogen_dioxide,A.ozone,A.sulfur_dioxide,A.carbon_dioxide,'S&F_RURAL',A.TAU550,2019,6,15,12,54.767273,-1.568486,0)
# data = np.array([wavelenth*1e-9,irradiance*1e9]).T
# plt.plot(wavelenth,irradiance)
#header = 'gpvdm\ntitle Light intensity of AM 1.5 Sun\ntype xy\nx_mul 1.0\ny_mul 1000000000.0\nz_mul 1.0\ndata_mul 1.0\nx_label\ny_label Wavelength\nz_label \ndata_label Intensity\nx_units \ny_units nm\nz_units \ndata_units m^{-1}.W.m(^-2)\nlogy False\nlogx False\nlogz False\ntime 0.0\nVexternal 0.0'
#np.savetxt(dir, data, delimiter='\t',header=header)

lat = [54.767273,31.519601,19422804,34.756545,18.371388]
lon = [-1.568486,74.370776,-99.129631,113.686424,-66.123175]
alt = [123e-3,217e-3,2240e-3,163e-3,8e-3]
t = [0, 5,-6, 8,-4]
label = ['Durham','Lahore','Mexico City','Zhengzhou','San Juan']
plot(lat,lon, alt,t,label)
plt.ylabel('Intensity (m$^{-1}$.W.m$^{-2}$)')
plt.xlabel('Wavelength (nm)')
plt.legend()
plt.show()
