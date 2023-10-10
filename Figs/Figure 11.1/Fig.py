import cartopy.crs as ccrs
import cartopy.util as ccut
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp2d
import datetime as dt
from scipy import stats
import matplotlib.colors as colors

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

def calc_multiyear(locations, start_date, end_date,noct):
    years = np.arange(start_date.year,end_date.year+1)
    data = []
    for location in locations:
        polluted = []
        for year in years:

            #pristine_dir = 'PERC_'+location+'_pristine_'+str(year)
            #polluted_dir = 'PERC_'+location+'_'+str(year)

            pristine_dir = 'FullRes_PERC_' + location + '_pristine_tilt_angle__NOCT_' + str(noct) + '_' + str(year)
            polluted_dir = 'FullRes_PERC_' + location + '_tilt_angle__NOCT_' + str(noct) + '_' + str(year)

            pristine_df = split_date(fetch_df(pristine_dir))
            polluted_df = split_date(fetch_df(polluted_dir))

            polluted_df['Energy_Loss'] = (((polluted_df['Pmax']*10)) - (pristine_df['Pmax']*10)) #/ (pristine_df['Power'] * 1e9) * 100
            polluted_df['Energy_Loss'] = polluted_df['Energy_Loss']#.clip(upper=0)

            polluted.append(polluted_df)
        polluted = pd.concat(polluted).fillna(0)
        data.append(polluted['Energy_Loss'].to_numpy())
    polluted.sort_index()
    return polluted.index, data

def running_mean(x, N):
   cumsum = np.cumsum(np.insert(x, 0, 0))
   return (cumsum[N:] - cumsum[:-N]) / N


def spacial(dir, start_date, end_date, step):

    start_date = pd.to_datetime(start_date,dayfirst=True,)
    end_date = pd.to_datetime(end_date,dayfirst=True)

    start_date = start_date.replace(hour=12)
    end_date = end_date.replace(hour=12)

    dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Location Lists',dir+'.csv')
    subfiles = pd.read_csv(dir)
    subfiles_names = [subfiles.loc[i]['Name'] +'_'+ subfiles.loc[i]['State'] for i in range(len(subfiles))]
    
    dates_of_data,energy_loss = calc_multiyear(subfiles_names,start_date,end_date,40)
    epoch = dt.datetime(1970,1,1)
    start_date_epoch = (start_date - epoch).total_seconds()
    end_date_epoch = (end_date - epoch).total_seconds()
    duration_seconds = end_date_epoch - start_date_epoch
    duration_minutes = duration_seconds/60
    duration_hours = duration_minutes/60
    duration_days = duration_hours/24

    fig,ax = plt.subplots()


    latitudes = subfiles['Latitude']
    longitudes = subfiles['Longitude']

    latitudes_unique = latitudes.unique()
    longitudes_unique = longitudes.unique()
    longitudes_unique = np.sort(longitudes_unique)


    latitudes = latitudes.values.ravel().reshape((len(latitudes_unique),len(longitudes_unique)))
    longitudes = longitudes.values.ravel().reshape((len(latitudes_unique),len(longitudes_unique)))
    longitudes_unique = subfiles['Longitude'].unique()
    longitudes_unique.sort()
    data = np.zeros((len(latitudes_unique),len(longitudes_unique),len(dates_of_data)))
    for idx,lon in enumerate(longitudes_unique):
        for jdx,lat in enumerate(latitudes_unique):
            index = subfiles.index[(subfiles['Latitude'] == lat)&(subfiles['Longitude'] == lon)].values[0]
            data[jdx,idx,:] = energy_loss[index]

    dates = pd.date_range(start_date,end_date,freq=str(step)+'H').values.ravel()
    dates_of_data_list = dates_of_data.values.ravel()
    dates_idx = np.searchsorted(dates_of_data_list,dates)

    #loss = np.zeros((len(dates_idx)-365,-1))
    std = np.zeros(len(dates_idx))
    #mn = np.zeros(len(dates_idx))
    #ma = np.zeros(len(dates_idx))
    #b = np.arange(-100, 10.5, 0.5)
    b = np.linspace(-100,10,50)
    #loss = np.zeros((len(dates_idx) - 365, len(b)))
    #for i in range(0, len(dates_idx)-365, 1095):
    d = data[:, :, dates_idx]
    d = np.reshape(d,(-1,1)).ravel()
    avg = np.mean(d)
    med = np.median(d)
        #loss.append(np.histogram(data[:,:,dates_idx[i:i+7]],bins=b,range=(-100,0),density=True)[0])
        #kern = stats.gaussian_kde(d)
        #l = kern(b)
    #plt.hist(d,b)
    #plt.axvline(avg,color='green')
    #plt.axvline(med,color='red')
        #plt.plot(b,l,label=str(int(2003 + (i/365))))
        #loss[i] = l
        #std[i] = np.std(data[:,:,dates_idx[i]])
        #mn[i] = np.min(data[:,:,dates_idx[i]])
        #ma[i] = np.max(data[:,:,dates_idx[i]])
        #loss[i] = np.percentile(data[:,:,dates_idx[i]],30)
    #loss = loss.T
    #loss = loss[loss != 0]
    #plt.pcolormesh(loss, cmap='inferno')
    #locs,labels = plt.yticks()
    #plt.yticks(range(len(b))[::10], np.arange(-100, 0, 2)[::10])
    plt.ylabel('Density')
    plt.ylim(bottom=0,top=0.04)
    plt.xlabel('Electrical Power loss (Wm$^{-2}$)')
    #plt.xticks(range(len(dates_idx)-365)[::730],np.arange(2003,2021,2))
    #plt.colorbar(label='Density')
    #plt.tight_layout()
    #plt.show()

    #c = np.argmax(counts)
    #print(c)
    counts, bins = np.histogram(d, bins=b)
    c = np.argmax(counts)
    return counts, bins, d

counts_2010, bins_2010, d_2010 = spacial('Beijing', '1/01/2010', '31/12/2010', 24)
counts_2020, bins_2020, d_2020 = spacial('Beijing', '1/01/2020', '31/12/2020', 24)

kde_2010 = stats.gaussian_kde(d_2010)
kde_2020 = stats.gaussian_kde(d_2020)
xx = np.linspace(-130,30,10000)
plt.plot(xx,kde_2010(xx))
plt.plot(xx,kde_2020(xx))
#plt.plot(bins_2020[:-1], counts_2020)
#plt.plot(bins_2010[:-1], counts_2010)
#plt.bar(bins_2020[:-1], counts_2020, width=2)
#plt.show()

# c_2010 = []
# for i in range(2010,2021,10):
#     counts, bins = spacial('Beijing', '1/01/'+str(i), '31/12/'+str(i), 24)
#     c_2010 = counts
    #plt.xlabel('Electrical Power loss (Wm$^{-2}$)')
    #plt.ylabel('Count')
    #plt.ylim(top=5000)
    #plt.savefig('Figure_11_W_' + str(i) + '.png', dpi=600)
    #plt.clf()
#plt.legend()
plt.savefig('Figure_11_kde.png',dpi=600)
#plt.savefig('Figure_111_2003.png',dpi=600)
#plt.savefig('Caribbean_Spacial_2020.png',dpi=600)
#plt.savefig('California_Spacial.svg')