import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from smarts import *

def calc_avg_spec(A,start_date,end_date):
    fileA = os.path.join(os.getcwd(),'Results',A+'.csv')
    dA = pd.read_csv(fileA)
    mask = (dA['Date'] > start_date) & (dA['Date'] <= end_date)
    dA = dA.loc[mask]
    dA = dA.mean(axis=0)
    print(dA)
    wavelegth, intensity = spectrum(dA['Air Pressure'],0,dA['Air Temperature'],dA['Relative Humidity'],'SUMMER',dA['Air Temperature'],dA['Formaldehyde'],dA['Methane'],dA['Carbon Monoxide'],dA['Nitric Acid'],dA['Nitrogen Dioxide'],dA['Ozone'],dA['Sulfur Dioxide'],dA['Carbon Dioxide'],'S&F_RURAL',dA['TAU550'],dA['Water Vapour'],2020,9,15,12,38.491299,-122.772402,-6)
    plt.plot(wavelegth,intensity)
    return 

calc_avg_spec('P3HTPCBM_Santa_Rosa_2020','2020-08-31','2020-10-31')
plt.show()