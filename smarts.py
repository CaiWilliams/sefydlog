import os
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import secrets

class input_file:
    def __init__(self,name = 'smarts295'):
        self.name = name
        self.file_extension = '.inp.txt'
        #if os.getcwd() != 'SMARTS':
        #    os.chdir('SMARTS')
        self.hash = secrets.token_hex(nbytes=16)
        self.filename = os.path.join('SMARTS_'+str(self.hash),name.lower() + self.file_extension)

    def add_comment(self, comment):
        self.comment = comment.text
        return self
    
    def add_pressure(self, pressure):
        match pressure.option:
            case 0:
                self.pressure = str(0) + '\n' + str(pressure.surface_pressure) + '\n'
            case 1:
                self.pressure = str(1) + '\n' + str(pressure.surface_pressure) + ' ' + str(pressure.altitude) + ' ' + str(pressure.height) 
            case 2:
                self.pressure = str(2) + '\n' + str(pressure.latitude) + ' ' + str(pressure.altitude) + ' ' + str(pressure.height)
        return self
    
    def add_atmosphere(self, atmosphere):
        match atmosphere.option:
            case 0:
                self.atmosphere = str(0) + '\n' + str(atmosphere.atmospheric_site_temp) + ' ' + str(atmosphere.relative_humidity) + ' ' + str(atmosphere.season) + ' ' + str(atmosphere.average_daily_temp)
            case 1:
                self.atmosphere = str(1) + '\n' + '\'' + str(atmosphere.reference) + '\''
        return self
    
    def add_water_vapor(self, water_vapor):
        match water_vapor.option:
            case 0:
                self.water_vapor = str(0) + '\n' + str(water_vapor.water)
            case 1:
                self.water_vapor = str(1)
            case 2:
                self.water_vapor = str(2)
        return self

    def add_ozone(self, ozone):
        match ozone.option:
            case 0:
                self.ozone = str(0) + '\n' + str(ozone.altitude_correction) + ' ' + str(ozone.abundance)
            case 1:
                self.ozone = str(1)
        return self
    
    def add_gas(self, gas):
        match gas.option:
            case 0:
                if gas.load == 0:
                    self.gas = str(0) + '\n' + str(gas.load) + '\n' + str(gas.formaldehyde) + ' ' + str(gas.methane) + ' ' + str(gas.carbon_monoxide) + ' ' + str(gas.nitrous_acid) + ' ' + str(gas.nitric_acid) + ' ' + str(gas.nitric_oxide) + ' ' + str(gas.nitrogen_dioxide) + ' ' + str(gas.nitrogen_trioxide) + ' ' + str(gas.ozone) + ' ' + str(gas.sulfur_dioxide)
                else:
                    self.gas = str(0) + '\n' + str(gas.load)
            case 1:
                self.gas = str(1)
        return self
    
    def add_carbon_dioxide(self, carbon_dioxide):
        self.carbon_dioxide = str(carbon_dioxide.abundance) + '\n' + str(carbon_dioxide.spectrum)
        return self
    
    def add_aerosol(self, aerosol):
        if aerosol.model == 'USER':
            self.aerosol = '\'' + str(aerosol.model) + '\'' + '\n' + str(aerosol.alpha1) + ' ' + str(aerosol.alpha2) + ' ' + str(aerosol.omegl) + ' ' + str(aerosol.gg)
        else:
            self.aerosol = '\'' + str(aerosol.model) + '\''
        return self
    
    def add_turbidity(self, turbidity):
        self.turbidity = str(turbidity.option) + '\n' + str(turbidity.value)
        return self
    
    def add_abledo(self, abledo):
        if abledo.option == -1: 
            self.abledo = str(abledo.option) + '\n' + str(abledo.rhox)
        else:
            self.abledo = str(abledo.option)
        self.abledo = self.abledo + '\n' + str(abledo.tilt)
        if abledo.tilt == 1:
            self.abledo = self.abledo + '\n' + str(abledo.albdg) + ' ' + str(abledo.surface_angle) + ' '+  str(abledo.surface_azimuth)
            if abledo.albdg == -1:
                self.abledo = self.abledo + '\n' + str(abledo.rhog)
        return self
    
    def add_spectral_range(self, spectral_range):
        self.spectral_range = str(spectral_range.wavelength_min) + ' ' + str(spectral_range.wavelength_max) + ' ' + str(spectral_range.sun_correction) + ' ' +str(spectral_range.solar_constant)
        return self

    def add_print(self, print):
        self.print = str(print.option)
        if print.option >= 1:
            self.print = self.print + '\n' + str(print.wavelength_min) + ' ' + str(print.wavelength_max) + ' ' + str(print.interval)
        if print.option >= 2:
            self.print = self.print + '\n' + str(print.num_output_variabels) + '\n' + str(print.output_variabels[0])
            for v in print.output_variabels[1:]:
                self.print = self.print + ' ' + str(v)
        return self
    
    def add_circumsolar(self, circumsolar):
        self.circumsolar = str(circumsolar.option)
        if circumsolar.option == 1:
            self.circumsolar = self.circumsolar + '\n' + str(circumsolar.slope) + ' ' + str(circumsolar.aperture) + ' ' + str(circumsolar.limit)
        return self

    def add_scan(self, scan):
        self.scan = str(scan.option)
        if scan.option == 1:
            self.scan = self.scan + '\n' + str(scan.filter_shape) + ' ' + str(scan.wavelength_min) + ' ' + str(scan.wavelength_max) + ' ' + str(scan.step) + ' ' + str(scan.WidthHalfMaximum)
        return self

    def add_illuminance(self, illuminance):
        self.illuminance = str(illuminance.option)
        return self
    
    def add_ultra_violet(self, ultra_violet):
        self.ultra_violet = str(ultra_violet.option)
        return self

    def add_mass(self, mass):
        self.mass = str(mass.option)
        match mass.option:
            case 0:
                self.mass = self.mass + '\n' + str(mass.zenith) + ' ' + str(mass.azimuth)
            case 1:
                self.mass = self.mass + '\n' + str(mass.elevation) + ' ' + str(mass.azimuth)
            case 2: 
                self.mass = self.mass + '\n' + str(mass.air_mass)
            case 3:
                self.mass = self.mass + '\n' + str(mass.year) + ' ' + str(mass.month) + ' ' + str(mass.day) + ' ' + str(mass.hour) + ' ' + str(mass.latitude)+ ' ' + str(mass.longitude) + ' ' + str(mass.time_zone)
            case 4:
                self.mass = self.mass + '\n' + str(mass.month) + ' ' + str(mass.latitude) + ' ' + str(mass.time_interval)
        return self

    def save(self):
        shutil.copytree('SMARTS','SMARTS_'+str(self.hash)) 
        self.order = [self.pressure, self.atmosphere, self.water_vapor, self.ozone, self.gas, self.carbon_dioxide, self.aerosol, self.turbidity, self.abledo, self.spectral_range, self.print, self.circumsolar, self.scan, self.illuminance, self.ultra_violet, self.mass]
        main_string = self.comment
        for item in self.order:
            main_string = main_string + '\n' + item
        main_string = main_string + '\n'
        with open(self.filename, 'w') as f:
            f.write(main_string)
        return self
    
    def run(self):
        os.chdir('SMARTS_'+str(self.hash))
        os.system('./smarts295batch')
        os.chdir('..')
        return

    def retrive(self):
        data = pd.read_csv(os.path.join('SMARTS_'+str(self.hash),'smarts295.ext.txt'),delimiter=' ',index_col=False)
        return data['Wvlgth'],data['Global_tilted_irradiance']

    def plot(self):
        data = pd.read_csv(os.path.join('SMARTS','smarts295.ext.txt'),delimiter=' ',index_col=False)
        plt.plot(data['Wvlgth'],data['Global_tilted_irradiance'])

    def delete(self):
        shutil.rmtree(os.path.join(os.getcwd(),'SMARTS_'+str(self.hash)))
        #os.remove(os.path.join('SMARTS','smarts295.ext.txt'))
        #os.remove(os.path.join('SMARTS','smarts295.inp.txt'))
        #os.remove(os.path.join('SMARTS','smarts295.out.txt'))

class comment:
    def __init__(self, text):
        self.text = text
        self.text = self.text.replace(' ','_')
        self.text = '\''+self.text+'\''

class pressure:
    def __init__(self, option, **kwargs):
        self.option = option
        match option:
            case 0:
                self.surface_pressure = kwargs['surface_pressure']
            case 1:
                self.surface_pressure = kwargs['surface_pressure']
                self.altitude = kwargs['altitude']
                self.height = kwargs['height']
            case 2:
                self.latitude = kwargs['latitude']
                self.altitude = kwargs['altitude']
                self.heigth = kwargs['height']

class atmosphere:
    def __init__(self, option, **kwargs):
        self.option = option
        match option:
            case 0:
                self.atmospheric_site_temp = kwargs['atmospheric_site_temp']
                self.relative_humidity = kwargs['relative_humidity']

                seasons = ['WINTER','SUMMER']
                if kwargs['season'].upper() in seasons:
                    self.season = kwargs['season'].upper()
                elif kwargs['season'].upper() == 'FALL' or 'AUTUM':
                    self.season = 'WINTER'
                elif kwargs['season'].upper() == 'SPRING':
                    self.season = 'SUMMER'
                else:
                    self.season = 'SUMMER'
                
                self.average_daily_temp = kwargs['average_daily_temp']

                
            case 1:
                references = ['USSA','MLS','MLW','SAS','SAW','TRL','STS','STW','AS','AW']
                if kwargs['reference'].upper() in references:
                    self.reference = kwargs['reference'].upper()
                else:
                    self.reference = 'USSA'

class water_vapor:
    def __init__(self, option, **kwargs):
        self.option = option
        match option:
            case 0:
                self.water = kwargs['water']

class ozone:
    def __init__(self, option, **kwargs):
        self.option = option
        match option:
            case 0:
                self.altitude_correction = kwargs['altitude_correction']
                self.abundance = kwargs['abundance']

class gas:
    def __init__(self, option, **kwargs):
        self.option = option
        match option:
            case 0:
                self.load = kwargs['load']
                if self.load == 0:
                    self.formaldehyde = kwargs['formaldehyde']
                    self.methane = kwargs['methane']
                    self.carbon_monoxide = kwargs['carbon_monoxide']
                    self.nitrous_acid = kwargs['nitrous_acid']
                    self.nitric_acid = kwargs['nitric_acid']
                    self.nitric_oxide = kwargs['nitric_oxide']
                    self.nitrogen_dioxide = kwargs['nitrogen_dioxide']
                    self.nitrogen_trioxide = kwargs['nitrogen_trioxide']
                    self.ozone = kwargs['ozone']
                    self.sulfur_dioxide = kwargs['sulfur_dioxide']
            
class carbon_dioxide:
    def __init__(self, abundance, spectrum):
        self.abundance = abundance
        spectrums = [-1,0,1,2,3,4,5,6,7,8]
        if spectrum in spectrums:
            self.spectrum = spectrum
        else:
            self.spectrum = 0

class aerosol:
    def __init__(self, model, **kwargs):
        models = ['S&F_RURAL', 'S&F_URBAN','S&F_MARIT','S&F_TROPO','SRA_CONTL','SRA_URBAN','SRA_MARIT','B&D_C','B&D_C1','DESERT_MIN','DESERT_MAX','USER']
        if model.upper() in models:
            self.model = model.upper()
        else:
            self.model = 'S&F_RURAL'
        
        if self.model == 'USER':
            self.alpha1 = kwargs['alpha1']
            self.alpha2 = kwargs['alpha2']
            self.omegl = kwargs['omegl']
            self.gg  = kwargs['gg']

class turbidity:
    def __init__(self, option, **kwargs):
        self.option = option
        match option:
            case 0:
                self.value = kwargs['TAU5']
            case 1:
                self.value = kwargs['BETA']
            case 2:
                self.value = kwargs['BCHUEP']
            case 3:
                self.value = kwargs['RANGE']
            case 4:
                self.value = kwargs['VISI']
            case 5:
                self.value = kwargs['TAU550']

class abledo:
    def __init__(self, option, **kwargs):
        self.option = option
        match option:
            case -1:
                self.rhox = kwargs['RHOX']
        self.tilt = kwargs['tilt']
        if self.tilt == 1:
            self.albdg = kwargs['albdg']
            self.surface_angle = kwargs['surface_angle']
            self.surface_azimuth = kwargs['surface_azimuth']
            if self.tilt == 1 and self.albdg == -1:
                self.rhog = kwargs['rhog']

class spectral_range:
    def __init__(self, wavelength_min, wavelength_max, sun_correction, solar_constant):
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.sun_correction = sun_correction
        self.solar_constant = solar_constant

class print_output:
    def __init__(self, option, **kwargs):
        self.option = option
        if option >= 1:
            self.wavelength_min = kwargs['wavelength_min']
            self.wavelength_max = kwargs['wavelength_max']
            self.interval = kwargs['interval']
        if option >= 2:
            self.num_output_variabels = kwargs['num_output_variabels']
            self.output_variabels = kwargs['output_variables']

class circumsolar:
    def __init__(self, option, **kwargs):
        self.option = option
        if option == 1:
            self.slope = kwargs['slope']
            self.aperture = kwargs['aperture']
            self.limit = kwargs['limit']

class scan:
    def __init__(self, option, **kwargs):
        self.option = option
        if option == 1: 
            self.filter_shape = kwargs['filter_shape']
            self.wavelength_min = kwargs['wavelength_min']
            self.wavelength_max = kwargs['wavelength_max']
            self.step = kwargs['step']
            self.WidthHalfMaximum = kwargs['WidthHalfMaximum']

class illuminance:
    def __init__(self, option):
        self.option = option

class ultra_violet:
    def __init__(self, option):
        self.option = option

class mass:
    def __init__(self, option, **kwargs):
        self.option = option
        match option:
            case 0:
                self.zenit = kwargs['zenit']
                self.azimuth = kwargs['azimuth']
            case 1:
                self.elevation = kwargs['elevation']
                self.azimuth = kwargs['azimuth']
            case 2:
                self.air_mass = kwargs['air_mass']
            case 3:
                self.year = kwargs['year']
                self.month = kwargs['month']
                self.day = kwargs['day']
                self.hour = kwargs['hour']
                self.latitude = kwargs['latitude']
                self.longitude = kwargs['longitude']
                self.time_zone = kwargs['time_zone']
            case 4:
                self.month = kwargs['month']
                self.latitude = kwargs['latitude']
                self.time_interval = kwargs['time_interval']

def spectrum_pristine(surface_pressure, altitude, site_temp, relative_humidity, season, average_daily_temperature, aerosol_model, year, month, day, hour, latitude, longitude, timezone):
    A = comment('')
    B = input_file()
    B = B.add_comment(A)

    A = pressure(1, surface_pressure=surface_pressure, altitude=altitude,height=0)
    B = B.add_pressure(A)

    A = atmosphere(0,atmospheric_site_temp=site_temp, relative_humidity=relative_humidity,season=season, average_daily_temp=average_daily_temperature)
    B = B.add_atmosphere(A)

    A = water_vapor(1)
    B = B.add_water_vapor(A)

    A = ozone(1)
    B = B.add_ozone(A)

    A = gas(1, load=1)
    B = B.add_gas(A)

    A = carbon_dioxide(280,1)
    B = B.add_carbon_dioxide(A)

    A = aerosol(aerosol_model)
    B = B.add_aerosol(A)

    A = turbidity(5, TAU550=0)
    B = B.add_turbidity(A)

    # if latitude >= 0:
    #     az = 0
    #     tilt = latitude * 0.9 + 29
    # else:
    #     az = 0
    #     tilt = latitude * 0.9 - 23.5
    # A = abledo(38, tilt = 1, albdg=38, surface_angle=tilt, surface_azimuth=az)
    A  = abledo(38, tilt=0)
    #A = abledo(20, tilt = 1, albdg=20, surface_angle=37, surface_azimuth=90)
    B = B.add_abledo(A)

    A = spectral_range(300,3000,1,1367.0)
    B = B.add_spectral_range(A)

    A = print_output(2, wavelength_min = 280, wavelength_max = 4000, interval = 2, num_output_variabels=4, output_variables=[8, 9 ,10, 30])
    B = B.add_print(A)

    A = circumsolar(0, slope=0, aperture=2.9, limit=0)
    B = B.add_circumsolar(A)

    A = scan(0)
    B = B.add_scan(A)

    A = illuminance(0)
    B = B.add_illuminance(A)

    A = ultra_violet(0)
    B = B.add_ultra_violet(A)

    A = mass(3, year=year, month=month, day=day, hour=hour, latitude=latitude, longitude=longitude, time_zone=timezone)
    B = B.add_mass(A)
    
    B.save()
    B.run()
    try:
        wavelength, irradiance = B.retrive()
        B.delete()
        return wavelength, irradiance
    except:
        B.delete()

def spectrum_refrence_ozone(surface_pressure, altitude, site_temp, relative_humidity, season, average_daily_temperature, formaldehyde, methane, carbon_monoxide, nitric_acid, nitrogen_dioxide, ozone3, sulfur_dioxide, carbon_dioxide_ab, aerosol_model, TAU550, water_vapour, year, month, day, hour, latitude, longitude, timezone):
        A = comment('')
        B = input_file()
        B = B.add_comment(A)

        A = pressure(1, surface_pressure=surface_pressure, altitude=altitude,height=0)
        B = B.add_pressure(A)

        print(site_temp)
        A = atmosphere(0,atmospheric_site_temp=site_temp, relative_humidity=relative_humidity,season=season, average_daily_temp=average_daily_temperature)
        B = B.add_atmosphere(A)

        if water_vapour > 12:
            water_vapour = 12
        A = water_vapor(1)#, water=water_vapour)
        B = B.add_water_vapor(A)

        A = ozone(1)
        B = B.add_ozone(A)

        A = gas(0, load=0, formaldehyde=formaldehyde, methane=methane, carbon_monoxide=carbon_monoxide, nitrous_acid=0, nitric_acid=nitric_acid, nitric_oxide=0, nitrogen_dioxide=nitrogen_dioxide, nitrogen_trioxide=0, ozone=ozone3, sulfur_dioxide=sulfur_dioxide)
        B = B.add_gas(A)

        A = carbon_dioxide(carbon_dioxide_ab,1)
        B = B.add_carbon_dioxide(A)

        A = aerosol(aerosol_model)
        B = B.add_aerosol(A)

        A = turbidity(5, TAU550=TAU550)
        B = B.add_turbidity(A)

        if latitude >= 0:
            az = 0
            tilt = latitude * 0.9 + 29
        else:
            az = 0
            tilt = latitude * 0.9 - 23.5
            
        A = abledo(38, tilt = 0)
        #A = abledo(38, tilt = 1, albdg=38, surface_angle=tilt, surface_azimuth=az)
        #A = abledo(20, tilt = 1, albdg=20, surface_angle=37, surface_azimuth=90)
        B = B.add_abledo(A)

        A = spectral_range(300,3000,1,1367.0)
        B = B.add_spectral_range(A)

        A = print_output(2, wavelength_min = 280, wavelength_max = 4000, interval = 2, num_output_variabels=4, output_variables=[8, 9 ,10, 30])
        B = B.add_print(A)

        A = circumsolar(0, slope=0, aperture=2.9, limit=0)
        B = B.add_circumsolar(A)

        A = scan(0)
        B = B.add_scan(A)

        A = illuminance(0)
        B = B.add_illuminance(A)

        A = ultra_violet(0)
        B = B.add_ultra_violet(A)

        A = mass(3, year=year, month=month, day=day, hour=hour, latitude=latitude, longitude=longitude, time_zone=timezone)
        B = B.add_mass(A)
        
        B.save()
        B.run()
        try:
            wavelength, irradiance = B.retrive()
            B.delete()
            return wavelength, irradiance
        except:
            B.delete()

def spectrum(surface_pressure, altitude, site_temp, relative_humidity, season, average_daily_temperature, formaldehyde, methane, carbon_monoxide, nitric_acid, nitrogen_dioxide, ozone3, sulfur_dioxide, carbon_dioxide_ab, aerosol_model, TAU550, water_vapour, year, month, day, hour, latitude, longitude, timezone):
        A = comment('')
        B = input_file()
        B = B.add_comment(A)

        A = pressure(1, surface_pressure=surface_pressure, altitude=altitude,height=0)
        B = B.add_pressure(A)

        A = atmosphere(0,atmospheric_site_temp=site_temp, relative_humidity=relative_humidity,season=season, average_daily_temp=average_daily_temperature)
        B = B.add_atmosphere(A)

        if water_vapour > 12:
            water_vapour = 12
        A = water_vapor(0, water=water_vapour)
        B = B.add_water_vapor(A)

        A = ozone(0,altitude_correction=0,abundance=ozone3*1e1)
        B = B.add_ozone(A)

        A = gas(0, load=0, formaldehyde=formaldehyde, methane=methane, carbon_monoxide=carbon_monoxide, nitrous_acid=0, nitric_acid=nitric_acid, nitric_oxide=0, nitrogen_dioxide=nitrogen_dioxide, nitrogen_trioxide=0, ozone=ozone3, sulfur_dioxide=sulfur_dioxide)
        B = B.add_gas(A)

        A = carbon_dioxide(carbon_dioxide_ab,1)
        B = B.add_carbon_dioxide(A)

        A = aerosol(aerosol_model)
        B = B.add_aerosol(A)

        A = turbidity(5, TAU550=TAU550)
        B = B.add_turbidity(A)

        if latitude >= 0:
            az = 0
            tilt = latitude * 0.9 + 29
        else:
            az = 0
            tilt = latitude * 0.9 - 23.5
            
        A = abledo(38, tilt = 0)
        #A = abledo(38, tilt = 1, albdg=38, surface_angle=tilt, surface_azimuth=az)
        #A = abledo(20, tilt = 1, albdg=20, surface_angle=37, surface_azimuth=90)
        B = B.add_abledo(A)

        A = spectral_range(300,3000,1,1367.0)
        B = B.add_spectral_range(A)

        A = print_output(2, wavelength_min = 280, wavelength_max = 4000, interval = 2, num_output_variabels=4, output_variables=[8, 9 ,10, 30])
        B = B.add_print(A)

        A = circumsolar(0, slope=0, aperture=2.9, limit=0)
        B = B.add_circumsolar(A)

        A = scan(0)
        B = B.add_scan(A)

        A = illuminance(0)
        B = B.add_illuminance(A)

        A = ultra_violet(0)
        B = B.add_ultra_violet(A)

        A = mass(3, year=year, month=month, day=day, hour=hour, latitude=latitude, longitude=longitude, time_zone=timezone)
        B = B.add_mass(A)
        
        B.save()
        B.run()
        try:
            wavelength, irradiance = B.retrive()
            B.delete()
            return wavelength, irradiance
        except:
            B.delete()

if __name__ == '__main__':
    X = np.linspace(0,100,10)
    for x in X:
        A = comment('Testting test testing')
        B = input_file()
        B = B.add_comment(A)

        A = pressure(1, surface_pressure=1013.5, altitude=0,height=0)
        B = B.add_pressure(A)

        A = atmosphere(1, reference='USSA')
        B = B.add_atmosphere(A)

        A = water_vapor(1)
        B = B.add_water_vapor(A)

        A = ozone(1)
        B = B.add_ozone(A)

        #A = gas(1)
        A = gas(0,load=0, formaldehyde=x, methane=0, carbon_monoxide=0, nitrous_acid=0, nitric_acid=0, nitric_oxide=0, nitrogen_dioxide=0, nitrogen_trioxide=0, ozone=0, sulfur_dioxide=0)
        B = B.add_gas(A)

        A = carbon_dioxide(370, 1)
        B = B.add_carbon_dioxide(A)

        A = aerosol('S&F_RURAL')
        B = B.add_aerosol(A)

        A = turbidity(0, TAU5 = 0.084)
        B = B.add_turbidity(A)

        A = abledo(20, tilt = 0)#, albdg=20, surface_angle=37, surface_azimuth=90)
        B = B.add_abledo(A)

        A = spectral_range(280,4000,1,1367.0)
        B = B.add_spectral_range(A)

        A = print_output(2, wavelength_min = 280, wavelength_max = 4000, interval = 2, num_output_variabels=4, output_variables=[8, 9 ,10, 30])
        B = B.add_print(A)

        A = circumsolar(1, slope=0, aperture=2.9, limit=0)
        B = B.add_circumsolar(A)

        A = scan(0)
        B = B.add_scan(A)

        A = illuminance(0)
        B = B.add_illuminance(A)

        A = ultra_violet(0)
        B = B.add_ultra_violet(A)

        A = mass(2, air_mass=1.5)
        B = B.add_mass(A)

        B.save()
        B.run()
        B.plot()
        B.delete()
    plt.show()