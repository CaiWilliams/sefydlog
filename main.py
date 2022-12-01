from smarts import * 
from gas_ppmv import *
from matplotlib.animation import FuncAnimation,PillowWriter


A = atmosphere_analysis('EAC4.grib','EGG4.grib',2021,11,29,0,54.767273,-1.568486)
A.get_gasses()
A.average_layers()
A.get_location()

def day(x):
    ax.clear()
    ax.set_ylim(bottom=0,top=1)
    ax.set_ylabel('Suns')
    ax.set_xlabel('Wavelength (nm)')
    try:
        wavelenth, irradiance = spectrum(A.surface_air_pressure,0,A.two_m_temperature,50,'WINTER',A.air_temperature,A.formaldehyde,A.methane,A.carbon_monoxide,A.nitric_acid,A.nitrogen_dioxide,A.ozone,A.sulfur_dioxide,A.carbon_dioxide,'S&F_RURAL',0.084,2021,11,29,12,54.767273,-1.568486,0)
    except:
        wavelenth = np.arange(280,4000,2)
        irradiance = np.zeros(len(wavelenth))
        p = ax.plot(wavelenth,irradiance)
        print('')
    return p

fig,ax = plt.subplots()
ani = FuncAnimation(fig,day, interval=0.02, blit=True, repeat=True, frames=480)
ani.save('DurhamDay.gif',dpi=300,writer=PillowWriter(fps=7))