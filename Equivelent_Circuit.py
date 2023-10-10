import os
import numpy as np
import pandas as pd
import pvlib.pvsystem
import scipy.constants as const
import matplotlib.pyplot as plt
from pvlib.pvsystem import i_from_v
from scipy.optimize import curve_fit


class cell:
    def __init__(self):
        return

    def load_eqe(self, dir_f):
        f = os.path.join(os.getcwd(),'EQE',dir_f+'.csv')
        df = pd.read_csv(f)
        wavelength = df['Wavelength'].to_numpy()
        eqe = df['EQE'].to_numpy()
        self.eqe = [wavelength, eqe]
        return

    def load_sepectrum(self, dir_f, scaling=1):
        f = os.path.join(os.getcwd(), 'Spectrums', dir_f+'.csv')
        df = pd.read_csv(f)
        wavelength = df['Wavelength'].to_numpy()
        intensity = df['Intensity'].to_numpy()
        #x = np.arange(wavelength[0],wavelength[-1]+0.5,0.5)
        #step = (x[-1] - x[0])/1000
        #i = np.interp(x,wavelength,intensity)
        half = np.argwhere((wavelength>=280)&(wavelength<400))
        one = np.argwhere((wavelength>=400)&(wavelength<1700))
        #five = np.argwhere((wavelength>=1700)&(wavelength<4000))

        i_half = intensity[half].ravel()
        x_half = wavelength[half].ravel()

        i_one = intensity[one].ravel()
        x_one = wavelength[one].ravel()
        x_one_half = np.around(np.arange(x_one[0],x_one[-1]+0.5,0.5),1).ravel()
        i_one_half = np.interp(x_one_half,x_one,i_one)/2

        # i_five = intensity[five].ravel()
        # x_five = wavelength[five].ravel()
        # x_five_half = np.around(np.arange(x_five[0],x_five[-1]+0.5,0.5),1).ravel()
        # i_five_half = np.interp(x_five_half,x_five,i_five)/10
        

        wavelength = np.concatenate((x_half,x_one_half))#,x_five_half))
        intensity = np.concatenate((i_half,i_one_half))#,i_five_half))

        self.spectrum = [wavelength, (intensity * scaling)* 100e-9] 
        self.incoming_power = np.cumsum((intensity * scaling))[-1]/10
        return
    
    def load_sepectrum_temp(self, idx, scaling=1):
        f = os.path.join(os.getcwd(), 'Temp', 'Spectrums', 'temp'+str(idx)+'.csv')
        df = pd.read_csv(f)
        wavelength = df['Wavelength'].to_numpy() * 1e9
        intensity = df['Intensity'].to_numpy() * 1e9
        if len(wavelength) < 2:
            return False
        #x = np.arange(wavelength[0],wavelength[-1]+0.5,0.5)
        #step = (x[-1] - x[0])/1000
        #i = np.interp(x,wavelength,intensity)
        half = np.argwhere((wavelength>=280)&(wavelength<400))
        one = np.argwhere((wavelength>=400)&(wavelength<1700))
        #five = np.argwhere((wavelength>=1700)&(wavelength<4000))

        i_half = intensity[half].ravel()
        x_half = wavelength[half].ravel()

        i_one = intensity[one].ravel()
        x_one = wavelength[one].ravel()
        x_one_half = np.around(np.arange(x_one[0],x_one[-1]+0.5,0.5),1).ravel()
        i_one_half = np.interp(x_one_half,x_one,i_one)/2

        # i_five = intensity[five].ravel()
        # x_five = wavelength[five].ravel()
        # x_five_half = np.around(np.arange(x_five[0],x_five[-1]+0.5,0.5),1).ravel()
        # i_five_half = np.interp(x_five_half,x_five,i_five)/10
    

        wavelength = np.concatenate((x_half,x_one_half))#,x_five_half))
        intensity = np.concatenate((i_half,i_one_half))#,i_five_half))
        self.spectrum = [wavelength, (intensity * scaling)* 100e-9] 
        self.incoming_power = np.cumsum((intensity * scaling))[-1]/10
        return True

    def load_jv(self, dir_f):
        f = os.path.join(os.getcwd(), 'JV', dir_f+'.csv')
        df = pd.read_csv(f)
        voltage = df['Voltage'].to_numpy()
        current = df['Current'].to_numpy()
        self.jv = [voltage, current]
        return

    def calculate_resistance(self, slope_frac=0.05):
        x = self.jv[0]
        y = -self.jv[1]
        length = len(x)
        values = int(length * slope_frac)
        start = np.arange(0,values)
        end = np.arange(length-values,length)
        self.serise_resistance = np.abs(y[start[0]] - y[start[-1]] / x[start[0]] - x[start[-1]])
        self.shunt_resistance = np.abs(y[end[0]] - y[end[-1]] / x[end[0]] - x[end[-1]])
        return

    def load_dark_jv(self, dir_f):
        f = os.path.join(os.getcwd(), 'Dark JV', dir_f+'.csv')
        df = pd.read_csv(f)
        voltage = df['Voltage'].to_numpy()
        current = df['Current'].to_numpy()
        self.dark_jv = [voltage, current]
        return

    def calculate_local_ideality(self):
        y = self.dark_jv[1]
        self.dark_satuartion = self.dark_jv[1][0]
        y = np.log(y)
        y = np.diff(y)/np.diff(self.dark_jv[0])
        y = const.e / (y * const.k * 293.15)
        y = np.where(y > 2, np.nan, y)
        y = np.where(y < 0, np.nan, y)
        not_nans = np.argwhere(np.isnan(y) == False).ravel()
        self.local_ideality = [self.dark_jv[0], y]
        self.average_ideality = np.average(y[not_nans])
        return

    def calculate_photogenerated(self):
        eqe_W_min = self.eqe[0][0]
        eqe_W_max = self.eqe[0][-1]

        spectrum_W_min = self.spectrum[0][0]
        spectrum_W_max = self.spectrum[0][-1]

        if eqe_W_min >= spectrum_W_min:
            W_min = eqe_W_min
        else:
            W_min = spectrum_W_min
        if eqe_W_max <= spectrum_W_max:
            W_max = eqe_W_max
        else:
            W_max = spectrum_W_max

        eqe_arg_W_min = np.argwhere(self.eqe[0][:] == W_min)[0][0]
        eqe_arg_W_max = np.argwhere(self.eqe[0][:] == W_max)[0][0]


        eqe = self.eqe[:][eqe_arg_W_min:eqe_arg_W_max]
        spectrum_arg_W_min = np.argwhere(self.spectrum[0][:] == W_min)[0][0]
        spectrum_arg_W_max = np.argwhere(self.spectrum[0][:] == W_max)[0][0]

        spectrum = [self.spectrum[0][spectrum_arg_W_min:spectrum_arg_W_max], self.spectrum[1][spectrum_arg_W_min:spectrum_arg_W_max]]
        if len(eqe[0]) < len(spectrum[0]):
            x = np.linspace(W_min, W_max, len(spectrum[0]))
            y = np.interp(x, eqe[0], eqe[1])
            eqe = [x, y]
        elif len(spectrum[0]) < len(eqe[0]):
            x = np.linspace(W_min, W_max, len(eqe))
            y = np.interp(x, spectrum[0], spectrum[1])
            spectrum = [x, y]
        I = np.zeros(len(eqe[0][:]))
        for j in range(len(eqe[0][:])):
            I[j] = eqe[1][j] * spectrum[1][j] * 0.5
        I = np.cumsum(I)
        I = I * (const.e / (const.h * const.c))

        self.photogenerated = I[-1]
        return
    
    def manual_entry(self, photogenerated, dark_saturation, shunt_resistance, serise_resistance, ideality):
        self.photogenerated = photogenerated
        self.dark_satuartion = dark_saturation
        self.shunt_resistance = shunt_resistance
        self.serise_resistance = serise_resistance
        self.average_ideality = ideality


    def equivilent_cuircuit_jv(self, air_temperature=293.15, NOCT=0):
        v = np.linspace(0, 1.5, 1000)
        temperature = (air_temperature-273.15) + ((((NOCT-273.15) - 20)/ 80) * self.incoming_power)
        temperature = temperature + 273.15
        #nNsVth = self.average_ideality * 1 * (const.k * temperature/const.e)
        i = i_from_v(resistance_shunt=self.shunt_resistance,resistance_series=self.serise_resistance,nNsVth=self.nNsVth,voltage=v,saturation_current=self.dark_satuartion, photocurrent=self.photogenerated)
        self.jv_experiment = [v,i]
        Jsc_arg = np.argmin(np.abs(v-0))
        self.Jsc = i[Jsc_arg]
        Voc_arg = np.argmin(np.abs(i-0))
        self.Voc = v[Voc_arg]
        return
    
    def func(self, v, shunt_resistance, serise_resistance, n):
        nNsVth = n * 1 * (const.k * 293.15/const.e)
        return i_from_v(resistance_shunt=shunt_resistance,resistance_series=serise_resistance,nNsVth=nNsVth,voltage=v,saturation_current=self.dark_satuartion, photocurrent=self.photogenerated)


    def fit_func(self):
        popt, pcov = curve_fit(self.func, self.jv[0], self.jv[1], maxfev=1000,)
        self.shunt_resistance = popt[0]
        self.serise_resistance = popt[1]
        self.n = popt[2]
        return

    def fit_func_pvlib(self):
        #self.jv[0],self.jv[1] = pvlib.ivtools.utils.rectify_iv_curve(self.jv[0],self.jv[1])
        Voc_arg = np.argmin(np.abs(self.jv[1] - 0))
        Voc = self.jv[0][Voc_arg]
        Jsc_arg = np.argmin(np.abs(self.jv[0] - 0))
        Jsc = self.jv[1][Jsc_arg]
        self.photogenerated,self.dark_satuartion,self.serise_resistance,self.shunt_resistance,self.nNsVth = pvlib.ivtools.sde.fit_sandia_simple(self.jv[0], self.jv[1],v_oc=Voc,i_sc=Jsc)
        self.serise_resistance = np.abs(self.serise_resistance)
        return

    def calculate_power(self):
        v = self.jv_experiment[0]
        i = self.jv_experiment[1]
        p = v * i
        self.arg_Pmax = np.argmax(p)
        self.Pmax = p[self.arg_Pmax]
        self.FF = self.Pmax/(self.Voc * self.Jsc)
        self.Efficiency = (self.Jsc * self.Voc * self.FF)/self.incoming_power
        return
    
def run_equivilent_circuit(idx,temperature,NOCT):
    Perc = cell()
    Perc.load_eqe('PERC')
    Perc.load_jv('PERC-1')
    Perc.calculate_resistance(slope_frac=0.1)
    Perc.load_dark_jv('PERC-1')
    Perc.calculate_local_ideality()
    if Perc.load_sepectrum_temp(idx,scaling=1) == False:
        return 0, 0, 0, 0, 0
    Perc.fit_func_pvlib()
    Perc.calculate_photogenerated()
    Perc.equivilent_cuircuit_jv(air_temperature=temperature,NOCT=NOCT)
    Perc.calculate_power()
    return Perc.Efficiency, Perc.Pmax, Perc.FF, Perc.Voc, Perc.Jsc

        
if __name__ == '__main__':
    temperature = 293.15
    NOCT = 293.15 + 40
    Perc = cell()
    Perc.load_eqe('PERC')
    Perc.load_jv('PERC-1')
    Perc.calculate_resistance(slope_frac=0.1)
    Perc.load_dark_jv('PERC-1')
    Perc.calculate_local_ideality()
    Perc.load_sepectrum('AM1.5G',scaling=1)
    #Perc.load_sepectrum_temp(0, scaling=2)
    Perc.fit_func_pvlib()
    Perc.calculate_photogenerated()
    Perc.equivilent_cuircuit_jv(air_temperature=temperature, NOCT=NOCT)
    Perc.calculate_power()
    plt.plot(Perc.jv_experiment[0],Perc.jv_experiment[1])
    plt.plot(Perc.jv[0],Perc.jv[1])
    #plt.ylim(bottom=-10,top=10)
    plt.show()

#Perc.load_eqe('D18PMIFFPMI')