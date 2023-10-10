
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import os
import difflib
from scipy.interpolate import interp2d
import datetime as dt

def fetch_df(name):
    file = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Results',name+'.csv')
    data = pd.read_csv(file)
    return data

def split_date(df):
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df = df.set_index('Date')
    df.loc[:,'Hour'] = df.index.hour.values
    df.loc[:,'Day'] = df.index.day.values
    df.loc[:,'Month'] = df.index.month.values
    return df

def calc_multiyear(locations, n, start_date, end_date,noct):
    sheet_names = ['Carbon Monoxide', 'Sulfur Dioxide', 'Nitric Acid', 'Water Vapour', 'Methane', 'TAU550', 'Formaldehyde', 'Air Temperature', 'Nitrogen Dioxide', 'Ozone', 'Carbon Dioxide', 'Air Pressure','Relative Humididity']
    if n == 'surface_air_pressure':
        name = 'Air Pressure'
    else:
        name = difflib.get_close_matches(n, sheet_names, 3)[0]
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:

            #pristine_dir = 'PERC_'+location+'_'+str(year)+'_relative_humidity'
            #pristine_dir = 'PERC_'+location+'_'+str(year)
            #polluted_dir = 'PERC_'+location+'_'+str(year)+'_'+n
            pristine_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year) #+ '_relative_humidity'
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year) + '_'+n

            polluted_dir_og = 'PERC_' + location + '_' + str(year)

            pristine_df = split_date(fetch_df(pristine_dir)) 
            polluted_df = split_date(fetch_df(polluted_dir))
            polluted_df_og = split_date(fetch_df(polluted_dir_og))

            polluted_df['Energy_Loss'] = ((((polluted_df['Pmax']*10)) - (pristine_df['Pmax']*10)))#/ (pristine_df['Power'] * 1e9) * 100) #/ polluted_df[name]# - pristine_df[name])
            #polluted_df_og['Energy_Loss'] = (((polluted_df_og['Pmax']*10)) - (pristine_df['Pmax']*10)) / (pristine_df['Power'] * 1e9)
            #print(polluted_df['Energy_Loss'])
            #print(polluted_df_og['Energy_Loss'])
            #polluted_df['Energy_Loss'] = (polluted_df['Energy_Loss'] - polluted_df_og['Energy_Loss']) / polluted_df_og['Energy_Loss']

            polluted.append(polluted_df)
        polluted = pd.concat(polluted)
        polluted = polluted.fillna(0)
        polluted = polluted.interpolate('index').rolling('7D',center=True).mean()
        mask = (polluted.index >= start_date) & (polluted.index <= end_date)
        polluted = polluted[mask]
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data


def pi(loc,start_date,end_date,n):
    #n = ['nitric_acid']
    #n = ['carbon_monoxide', 'sulfur_dioxide', 'nitric_acid', 'water_vapour', 'methane', 'TAU550', 'formaldehyde', 'air_temperature', 'nitrogen_dioxide', 'ozone', 'carbon_dioxide', 'surface_air_pressure','relative_humidity']
    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True,)

    data = []
    dates, temp_data = calc_multiyear([loc],n,start_date,end_date,40)
    l = len(dates)
    data.append(temp_data[:])
    data = np.asarray(data).reshape(1,l)

    return n,data

def loop(locs,start_date,end_date):
    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True,)
    dates = pd.date_range(start_date,end_date,freq='3H')
    n = ['relative_humidity']
    sheet_names = ['Carbon Monoxide']

    sd = str(start_date.day) + '/' + str(start_date.month) + '/' + str(start_date.year)
    ed = str(end_date.day) + '/' + str(end_date.month) + '/' + str(end_date.year)

    data = []
    for idx,_n in enumerate(n):
        for loc in locs:
            n,d = pi(loc,sd,ed,_n)
            data.append(d[0])
            #plt.plot(dates,d[0])
    data_base = np.asarray(data)

    n = ['TAU550']
    sheet_names = ['TAU550']

    sd = str(start_date.day) + '/' + str(start_date.month) + '/' + str(start_date.year)
    ed = str(end_date.day) + '/' + str(end_date.month) + '/' + str(end_date.year)

    data = []
    for idx, _n in enumerate(n):
        for loc in locs:
            n, d = pi(loc, sd, ed, _n)
            data.append(d[0])
            # plt.plot(dates,d[0])
    data = np.asarray(data)
    #print(data)
    #n = ['TAU550']
    #plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.tab20c.colors)
    # for idx in range(len(data[0])):
    #     s = np.sum(data[:,idx])
    #     print(data[:,idx])
    #     print(s)
    #     data[:,idx] = data[:,idx]/s
    print("Mean: ", np.mean(data[:,:]-data_base[:,:]))
    for idx in range(len(data)):
        plt.plot(dates, data[idx,:]-data_base[idx,:])#, c='tab:blue')
        plt.xticks([dt.datetime(2020,8,1,12),dt.datetime(2020,8,31,12),dt.datetime(2020,9,30,12)])
        #plt.xlim(left=dt.datetime(2020,8,1,12),right=dt.datetime(2020,9,30,12))
        plt.ylabel('Electrical Power Loss Recovered \n Removing ' + sheet_names[0] +' (Wm$^{-2}$)')
        #plt.ylim(top=1.5,bottom=0)
        #plt.yticks(ticks=range(len(data)),labels=n,fontsize=9)
        #plt.colorbar()
        #plt.tight_layout()
        #plt.yscale('log')
        #plt.savefig(locs[0]+'_'+n[idx]+'.png', dpi=600)
        #plt.show()
        #plt.clf()
    #plt.show()
    plt.savefig('Figure_041_'+n+'.png',dpi=600)

loop(['Santa_Clarita_California','Fresno_California','Sacramento_California','Redding_California'],'1/08/2020','30/09/2020')
#plt.savefig()
#plt.show()
#loop(['Santa_Clarita_California'],'16/08/2020','12/11/2020')
