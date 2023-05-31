
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import os
import datetime as dt

def fetch_df(name):
    file = os.path.join(os.path.dirname(os.getcwd()),'Results',name+'.csv')
    data = pd.read_csv(file)
    return data

def split_date(df):
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df = df.set_index('Date')
    df.loc[:,'Hour'] = df.index.hour.values
    df.loc[:,'Day'] = df.index.day.values
    df.loc[:,'Month'] = df.index.month.values
    return df

def calc_multiyear(locations, start_date, end_date,noct):
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:
            pristine_dir = 'PERC_' + location + '_pristine_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            polluted_dir = 'PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            #pristine_dir = 'PERC_'+location+'_pristine_tilt_angle'+str(year)
            #polluted_dir = 'PERC_'+location+'_tilt_angle_'+str(year)

            pristine_df = split_date(fetch_df(pristine_dir)) 
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = ((polluted_df['Pmax']*10) - (pristine_df['Pmax']*10))
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

    dir = os.path.join(os.path.dirname(os.getcwd()),'Location Lists',dir+'.csv')
    subfiles = pd.read_csv(dir)
    subfiles_names = [subfiles.loc[i]['Name'] +'_'+ subfiles.loc[i]['State'] for i in range(len(subfiles))]

    locs_dir = os.path.join(os.path.dirname(os.getcwd()),'Location Lists',locs+'.csv')
    locs = pd.read_csv(locs_dir)
    locs_names = locs['Name'].to_numpy()[::]
    locs_lat = locs['Latitude'].to_numpy()
    locs_lon = locs['Longitude'].to_numpy()
    

    dates_of_data,energy_loss = calc_multiyear(subfiles_names,start_date,end_date,40)
    epoch = dt.datetime(1970,1,1)
    start_date_epoch = (start_date - epoch).total_seconds()
    end_date_epoch = (end_date - epoch).total_seconds()
    duration_seconds = end_date_epoch - start_date_epoch
    duration_minutes = duration_seconds/60
    duration_hours = duration_minutes/60
    duration_days = duration_hours/24

    # latitudes = subfiles['Latitude']
    # longitudes = subfiles['Longitude']
    #
    # latitudes_unique = latitudes.unique()
    # longitudes_unique = longitudes.unique()
    #
    # locs_lat_arg = [np.abs(latitudes_unique - lat).argmin() for lat in locs_lat][::-1]
    # locs_lon_arg = [np.abs(longitudes_unique - lon).argmin() for lon in locs_lon][::-1]
    #
    # latitudes = latitudes.values.ravel().reshape((len(longitudes_unique),len(latitudes_unique)))
    # longitudes = longitudes.values.ravel().reshape((len(longitudes_unique),len(latitudes_unique)))

    # data = np.zeros((len(longitudes_unique),len(latitudes_unique),len(dates_of_data)))
    # for idx,lon in enumerate(longitudes_unique):
    #     for jdx,lat in enumerate(latitudes_unique):
    #         index = subfiles.index[(subfiles['Latitude'] == lat)&(subfiles['Longitude'] == lon)].values[0]
    #         data[idx,jdx,:] = energy_loss[index]


    dates = pd.date_range(start_date,end_date,freq=str(step)+'H').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list,dates)
    print(np.shape(energy_loss))
    plt.rcParams["figure.figsize"] = (11, 7)
    fig, ax = plt.subplots(nrows=len(locs_names))
    colours = ['tab:blue','tab:orange','tab:green','tab:red'][::-1]
    for i in range(len(locs_names))[::]:
        #energy_loss_sum = [data[locs_lon_arg[i],locs_lat_arg[i],j] for j in range(dates_idx[0],dates_idx[-1])]
        #energy_loss_sum_C = np.interp(np.linspace(0,1,num=len(range(dates_idx[0],dates_idx[-1]))*3), np.linspace(0,1,num=len(range(dates_idx[0],dates_idx[-1]))),energy_loss_sum)
        ax[i].plot(dates_of_data_list[::8],energy_loss[i][::8],color=colours[i])
        print(locs_names[i])
        #print(np.nansum(energy_loss_sum_C))
        #print(np.nanmean(energy_loss_sum))
        ax[i].set_ylim(top=0,bottom=-60)
        ax[i].set_xlim(left=dates_of_data_list[dates_idx[0]],right=dates_of_data_list[dates_idx[-1]])
        ax[i].text(0.01,0.05,locs_names[i].replace('_',' '),color=colours[i],transform=ax[i].transAxes)
        ax[i].axvspan(xmin=dt.datetime(2008,5,23,12),xmax=dt.datetime(2008,6,3,12),facecolor='tab:purple',alpha=0.25   )
        if i != len(locs_names)-1:
            dates = [dt.datetime(2008, 1, 1, 12), dt.datetime(2008, 12, 31, 12)]
            #dates = [dt.datetime(2008, 1, 1, 12), dt.datetime(2008, 7, 2, 12), dt.datetime(2008, 12, 31, 12)]
            ax[i].set_xticks(dates)
            ax[i].xaxis.set_visible(False)
        else:
            dates = [dt.datetime(2008, 1, 1, 12),dt.datetime(2008, 12, 31, 12)]#,dt.datetime(2008,12,31,12)]
            #dates = [dt.datetime(2008, 1, 1, 12), dt.datetime(2008, 7, 2, 12), dt.datetime(2008, 12, 31, 12)]
        #    #dates = [dt.datetime(2008,8,16,12),dt.datetime(2008,9,5,12),dt.datetime(2008,9,25,12)]
            ax[i].set_xticks(dates)
        
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.075, right=0.95, wspace=0.2, hspace=0.1)
    fig.text(0.02, 0.5, 'Electrical Power Loss (Wm$^{-2}$)', va='center', rotation='vertical')

    return



spacial('EuropeLocs','12/06/2008','24/06/2008','24','EuropeLocs')
plt.show()
#plt.savefig('California_Engery_Loss.png',dpi=600)
#plt.savefig('California_Engery_Loss.svg')
