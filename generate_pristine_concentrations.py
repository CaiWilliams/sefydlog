from smarts import *

import matplotlib.pyplot as plt
import numpy as np
import math

def gas_concentrations(Site_Pressure, Site_Temperature, pollution):
    PP0 = Site_Pressure / 1013.25
    PP0X2 = PP0 * PP0

    TT0 = Site_Temperature / 273.15

    CH2O = 0.003 * np.ones(len(Site_Temperature))
    CH4 = 1.31195 * (PP0 * 1.1245) * (TT0 ** 0.047283)
    CO = 0.31491 * (PP0 ** 2.6105) * np.exp(0.82546 * PP0 + 0.88437 * PP0X2)
    NH02 = 0.0001 * np.ones(len(Site_Temperature))
    NH03 = 3.6739e-4 * (PP0 ** 0.13568) / (TT0 ** 0.0714)
    A = 0.74039+2.4154*PP0
    B = 57.314*PP0
    NO = 1e-4 * np.minimum(A, B)
    A = 1.864+0.20314*PP0
    B = 41.693*PP0
    NO2 = 1e-4 * np.minimum(A, B)
    NO3 = 0.00005 * np.ones(len(Site_Temperature))
    SO2 = 1.114e-5 * (PP0 ** 0.81319) * np.exp(0.81356 + 3.0448 * PP0X2 -1.5652 * PP0X2 * PP0)


    CH4 = CH4 - 146e-3
    CO = CO - 152e-3
    NH03 = NH03 -1.2001e-3
    NO2 = NO2 + 62.5e6
    SO2 = SO2 - 31.2e-6
    Ab = [CH2O,CH4,CO,NH02,NH03,NO,NO2,NO3,SO2]
    #Ab = [a for a in Ab]

    pollution = pollution.lower()
    pollution = pollution.replace(' ','')
    match pollution:
        case 'pristine':
            Ap = [-0.003,0,-0.1,0,-9.9e-4,0,0,0,-4.9e-4,-0.007,0]
        case 'lightpollution':
            Ap = [0.001,0.2,0,0.0005,0.001,0.075,0.005,1e-5,0.023,0.01]
        case 'moderate':
            Ap = [0.007,0.3,0.35,0.002,0.005,0.2,0.02,5e-5,0.053,0.05]
        case 'severe':
            Ap = [0.01,0.4,9.9,0.01,0.012,0.5,0.2,2e-4,0.175,0.2]
    Ab = np.asarray([Ab[i] + Ap[i] for i in range(len(Ab))])
    return Ab

if __name__ == '__main__':
    #print(pristine_conditions(1000,293.15,'Pristine'))
    RW,RI = spectrum_pristine(1001,0,40,20,'SUMMER',40,'S&F_RURAL',2020,7,1,12,54.773525,-1.575864,0)
    #plt.plot(RW,RI)

    Gasses = gas_concentrations(1000,293.15,'pristine')
    t = 281.57664011764706-273.15
    TW,TI = spectrum_refrence_ozone(92495.07952941177,0,T,85.46066450588233,'SUMMER',T,0,1.7474728258823529,0.19,6e-05,0.001,0,0.001,280,'S&F_RURAL',0,13.90289656,2020,7,1,12,54.773525,-1.575864,0)
    plt.plot(TW,TI-RI)

    plt.show()