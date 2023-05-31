import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd

eqe_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'EQE','PERC.csv')
eqe = pd.read_csv(eqe_dir)
plt.plot(eqe['Wavelength'],eqe['EQE'])

plt.twinx()

NO2_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'SMARTS','Gases','Abs_NO2.dat')
NO2 = pd.read_csv(NO2_dir,delimiter='\t')
plt.plot(NO2['wvl_nm'],(NO2['xsec_220K']-NO2['xsec_220K'].min())/(NO2['xsec_220K'].max()-NO2['xsec_220K'].min()))

O3_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'SMARTS','Gases','Abs_O3UV.dat')
O3 = pd.read_csv(O3_dir,delimiter='\t')

plt.plot(O3['Wvl_nm'],(O3['Ref_228K']-O3['Ref_228K'].min())/(O3['Ref_228K'].max()-O3['Ref_228K'].min()))

plt.xlim(left=eqe['Wavelength'].min(),right=1090)
plt.show()