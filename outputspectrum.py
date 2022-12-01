from smarts import * 
from gas_ppmv import *
import numpy as np



A = atmosphere_analysis('EAC4_Year.grib','EGG4.grib',2020,11,29,0,54.767273,-1.568486)
A.get_gasses()
A.average_layers()
A.get_location()

dir='spectra.inp'
wavelenth, irradiance = spectrum(A.surface_air_pressure,0,A.two_m_temperature,50,'WINTER',A.air_temperature,A.formaldehyde,A.methane,A.carbon_monoxide,A.nitric_acid,A.nitrogen_dioxide,A.ozone,A.sulfur_dioxide,A.carbon_dioxide,'S&F_RURAL',0.084,2021,11,29,12,54.767273,-1.568486,0)
data = np.array([wavelenth,irradiance]).T
plt.plot(wavelenth,irradiance)
header = 'gpvdm\ntitle Light intensity of AM 1.5 Sun\ntype xy\nx_mul 1.0\ny_mul 1000000000.0\nz_mul 1.0\ndata_mul 1.0\nx_label\ny_label Wavelength\nz_label \ndata_label Intensity\nx_units \ny_units nm\nz_units \ndata_units m^{-1}.W.m(^-2)\nlogy False\nlogx False\nlogz False\ntime 0.0\nVexternal 0.0'
np.savetxt(dir, data, delimiter='\t',header=header)
plt.show()
