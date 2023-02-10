import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp2d
import datetime as dt

def fetch_df(name):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    data = pd.read_csv(file)
    return data

def split_date(df):
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df = df.set_index('Date')
    df.loc[:,'Hour'] = df.index.hour.values
    df.loc[:,'Day'] = df.index.day.values
    df.loc[:,'Month'] = df.index.month.values
    return df

def calc_multiyear(locations, start_date, end_date):
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:

            pristine_dir = 'PERC_'+location+'_pristine'+str(year)
            polluted_dir = 'PERC_'+location+'_'+str(year)

            pristine_df = split_date(fetch_df(pristine_dir)) 
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = (polluted_df['Pmax']*10) - (pristine_df['Pmax']*10)
            polluted.append(polluted_df)
        polluted = pd.concat(polluted)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data

def spacial(dir, start_date, end_date, step, locs):

    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)

    dir = os.path.join(os.getcwd(),'Location Lists',dir+'.csv')
    subfiles = pd.read_csv(dir)
    subfiles_names = [subfiles.loc[i]['Name'] +'_'+ subfiles.loc[i]['State'] for i in range(len(subfiles))]

    locs_dir = os.path.join(os.getcwd(),'Location Lists',locs+'.csv')
    locs = pd.read_csv(locs_dir)
    locs_names = locs['Name'].to_numpy()[::-1]
    locs_lat = locs['Latitude'].to_numpy()
    locs_lon = locs['Longitude'].to_numpy()
    

    dates_of_data,energy_loss = calc_multiyear(subfiles_names,start_date,end_date)
    epoch = dt.datetime(1970,1,1)
    start_date_epoch = (start_date - epoch).total_seconds()
    end_date_epoch = (end_date - epoch).total_seconds()
    duration_seconds = end_date_epoch - start_date_epoch
    duration_minutes = duration_seconds/60
    duration_hours = duration_minutes/60
    duration_days = duration_hours/24

    
    dates = pd.date_range(start_date,end_date,freq=str(step)+'H').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list,dates)

    fig, ax = plt.subplots(nrows=len(locs_names))
    colours = ['tab:blue','tab:orange','tab:green','tab:red'][::-1]

    for i in range(len(locs_names)):
        df = pd.DataFrame(index=dates_of_data,data=energy_loss[i])
        #df = df[df != 0]
        df = df.rolling(window=8*31).mean()
        print(df)
        #energy_loss_sum = [energy_loss[i] for j in range(dates_idx[0],dates_idx[-1],8)]
        ax[i].plot(df,color=colours[i])
        #ax[i].set_ylim(top=0, bottom=-80)
        ax[i].set_xlim(left=dates_of_data_list[dates_idx[0]],right=dates_of_data_list[dates_idx[-1]])
        ax[i].text(dates_of_data_list[dates_idx[0]+2],-135,locs_names[i],color=colours[i])
        
        if i != len(locs_names)-1:
            ax[i].xaxis.set_visible(False)
        #else:
            #dates = [dt.datetime(2020,8,16,12),dt.datetime(2020,9,5,12),dt.datetime(2020,9,25,12)]
        #    ax[i].set_xticks(dates_of_data)
        
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.90, wspace=0.05, hspace=0.05)
    fig.text(0.04, 0.5, 'Energy Lost (Wm$^{-2}$)', va='center', rotation='vertical')

    return
 
spacial('VenezuelaLocs','1/01/2003','31/12/2020',3,'VenezuelaLocs')
plt.show()
