import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')


data = pd.read_csv('PERC.csv')
plt.plot(data['Wavelength'], data['EQE']*100,label='PERC')
data = pd.read_csv('PM6Y6.csv')
plt.plot(data['Wavelength'], data['EQE']*100,label='PM6:Y6')
data = pd.read_csv('D18PMIFFPMI.csv')
plt.plot(data['Wavelength'], data['EQE']*100,label='D18:PMI-FF-PMI')
plt.ylabel('External Quantum Efficiency (%)')
plt.xlabel('Wavelength (nm)')
plt.xlim(left=320, right=1180)
plt.tight_layout()
#plt.legend()
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=False, ncol=3)
plt.savefig('eqe_PERC_PM6Y6_D18PMIFFPMI.png',dpi=600)