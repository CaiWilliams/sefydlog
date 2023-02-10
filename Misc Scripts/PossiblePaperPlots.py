import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime


def plot(name):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    data = pd.read_csv(file)
    data = data['Pmax'].to_numpy()
    plt.plot(np.cumsum(data[::]))
    return data

def fetch(name,label):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    Data = pd.read_csv(file)
    data = Data[label]
    x = np.arange(len(data))
    xvals = np.arange(len(data))
    yvals = np.interp(xvals,x,data.to_numpy())
    xdates = pd.date_range(Data['Date'].loc[0],Data['Date'].loc[len(data)-1],freq='3h')
    return xdates,yvals

def fetch_df(name):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    data = pd.read_csv(file)
    return data

def calc_power_diff(A, B, linestyle="-", colour='tab:blue'):
    fileA = os.path.join(os.getcwd(),'Results',A+'.csv')
    fileB = os.path.join(os.getcwd(),'Results',B+'.csv')
    dA = pd.read_csv(fileA)
    dB = pd.read_csv(fileB)
    data = np.cumsum(dB['Pmax'] - dA['Pmax'])
    x = np.arange(len(data))
    xvals = np.arange(len(data))
    yvals = np.interp(xvals,x,data.to_numpy())
    xdates = pd.date_range(dA['Date'].loc[0],dA['Date'].loc[len(dA)-1],freq='3h')
    return xdates,yvals

def split_date(df):
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df = df.set_index('Date')
    df.loc[:,'Hour'] = df.index.hour.values
    df.loc[:,'Day'] = df.index.day.values
    df.loc[:,'Month'] = df.index.month.values
    return df

def calc_multiyear(locs,start_year,end_year):
    years = np.arange(start_year,end_year+1)
    data = []
    for location in locs:
        Dates = []
        Losses = []
        for year in years:
                dir_p = 'PERC_'+location+'_pristine_'+str(year)
                dir = 'PERC_'+location+'_'+str(year)
                df_p = split_date(fetch_df(dir_p))
                df = split_date(fetch_df(dir))
                df = df[~((df.index.month == 2) & (df.index.day == 29))]
                df_p = df_p[~((df_p.index.month == 2) & (df_p.index.day == 29))]
                Dates.append(df)
                df = df.groupby(['Month','Day','Hour']).mean()
                df_p = df_p.groupby(['Month','Day','Hour']).mean()
                df['Loss'] = (df['Pmax']*10) - (df_p['Pmax']*10)
                Losses.append(df)
        Dates = pd.concat(Dates)
        Dates = Dates.index
        Losses = pd.concat(Losses)
        Losses.reset_index()
        Losses = Losses['Loss'].to_numpy()
        data.append(Losses)
    return Dates, np.asarray(data)

def plot_multiyear(locs,start_year,end_year,num_average_years,colour=''):

    #if start_year - num_average_years < 2003:
    #    return print('Number of years average too long for data, end point 2003')

    years = np.arange(start_year,end_year+1)
    for location in locs:

        # Averages = []
        # for year in years:
        #     for i in range(year-num_average_years,year):
        #         if i == 2009:
        #             pass
        #         dir_pristine = 'PERC_'+location+'_pristine_'+str(i)
        #         dir = 'PERC_'+location+'_'+str(i)
        #         Temp_df_p = split_date(fetch_df(dir_pristine))
        #         Temp_df = split_date(fetch_df(dir))

        #         if i == year-num_average_years:
        #             df = Temp_df
        #             df_p = Temp_df_p
        #         else:
        #             df = pd.concat([df,Temp_df])
        #             df_p = pd.concat([df_p,Temp_df_p])
        #     df = df[~((df.index.month == 2) & (df.index.day == 29))]
        #     df_p = df_p[~((df_p.index.month == 2) & (df_p.index.day == 29))]
        #     df = df.groupby(['Month','Day','Hour']).mean()
        #     df_p = df_p.groupby(['Month','Day','Hour']).mean()
        #     df['Loss'] = (df['Pmax']*10) - (df_p['Pmax']*10)
        #     Averages.append(df)

        Dates = []
        Losses = []
        for year in years:

            dir_p = 'PERC_'+location+'_pristine_'+str(year)
            dir = 'PERC_'+location+'_'+str(year)
            df_p = split_date(fetch_df(dir_p))
            df = split_date(fetch_df(dir))
            df = df[~((df.index.month == 2) & (df.index.day == 29))]
            df_p = df_p[~((df_p.index.month == 2) & (df_p.index.day == 29))]
            Dates.append(df)
            df = df.groupby(['Month','Day','Hour']).mean()
            df_p = df_p.groupby(['Month','Day','Hour']).mean()
            df['Loss'] = (((df['Pmax']*10) - (df_p['Pmax']*10)) / (df_p['Pmax']*10))*100
            Losses.append(df)

        #for i in range(len(years)):
        #    Losses[i] = Losses[i].subtract(Averages[i])
        
        data = pd.concat(Losses)
        Dates = pd.concat(Dates)
        Dates = Dates.index.values
        #data.loc[:,'Loss'] = np.cumsum(data['Loss'])
        data = data.reset_index()

        data.index = Dates
        data = data.fillna(0)
        data['Loss'].rolling(window=8*365*5,center=True).mean()
        
        if len(Dates) < len(data['Loss'].values):
            y = data['Loss'].values[:len(Dates)]
        else:
            y = data['Loss'].values
        if colour == '':
            plt.plot(y)
        else:
            plt.plot(y,color=colour)

    return Dates

def plot_multiyear_mean(locs,start_year,end_year,num_average_years,colour=''):
    y_mean = []

    if start_year - num_average_years < 2003:
        return print('Number of years average too long for data, end point 2003')

    years = np.arange(start_year,end_year+1)
    #y_mean = []
    for location in locs:

        Averages = []
        for year in years:
            for i in range(year-num_average_years,year):
                dir_pristine = 'PERC_'+location+'_pristine_'+str(i)
                dir = 'PERC_'+location+'_'+str(i)
                Temp_df_p = split_date(fetch_df(dir_pristine))
                Temp_df = split_date(fetch_df(dir))

                if i == year-num_average_years:
                    df = Temp_df
                    df_p = Temp_df_p
                else:
                    df = pd.concat([df,Temp_df])
                    df_p = pd.concat([df_p,Temp_df_p])
            df = df[~((df.index.month == 2) & (df.index.day == 29))]
            df_p = df_p[~((df_p.index.month == 2) & (df_p.index.day == 29))]
            df = df.groupby(['Month','Day','Hour']).mean()
            df_p = df_p.groupby(['Month','Day','Hour']).mean()
            df['Loss'] = df['Pmax'] - df_p['Pmax']
            Averages.append(df)

        Dates = []
        Losses = []
        for year in years:

            dir_p = 'PERC_'+location+'_pristine_'+str(year)
            dir = 'PERC_'+location+'_'+str(year)
            df_p = split_date(fetch_df(dir_p))
            df = split_date(fetch_df(dir))
            df = df[~((df.index.month == 2) & (df.index.day == 29))]
            df_p = df_p[~((df_p.index.month == 2) & (df_p.index.day == 29))]
            Dates.append(df)
            df = df.groupby(['Month','Day','Hour']).mean()
            df_p = df_p.groupby(['Month','Day','Hour']).mean()
            df['Loss'] = df['Pmax'] - df_p['Pmax']
            Losses.append(df)

        for i in range(len(years)):
            Losses[i] = Losses[i].subtract(Averages[i])
        
        data = pd.concat(Losses)
        Dates = pd.concat(Dates)
        Dates = Dates.index.values
        data.loc[:,'Loss'] = np.cumsum(data['Loss'])
        data = data.reset_index()
        
        if len(Dates) < len(data['Loss'].values):
            y = data['Loss'].values[:len(Dates)]
        else:
            y = data['Loss'].values
        y_mean.append(y)
    y_mean = np.asarray(y_mean)
    if colour == '':
        plt.plot(Dates, np.mean(y_mean,axis=0))
    else:
        print(y_mean)
        plt.plot(Dates,np.mean(y_mean,axis=0),color=colour)
        
    return Dates

    locations = ['Durham','Reykjavik','Montreal']
    years = np.arange(2008,2021)
    for location in locations:

        Y = []
        for year in years:
            Temp = np.zeros(8746)
            for i in range(year-2,year):
                X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(i),'PERC_'+location+'_'+str(i))
                Temp = Temp + Temp_Y[:8746]
            Y.append(Temp/len(range(year-2,year)))
        Y = np.asarray(Y)

        X2 = []
        Y2 = []
        for year in years:
            Temp_X2,Temp_Y2 = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Temp_X2 = Temp_X2[:8746]
            Temp_Y2 = Temp_Y2[:8746]
            X2.append(Temp_X2)
            Y2.append(Temp_Y2)
        
        #Y = Y/len(years)
        Y2 = np.asarray(Y2).reshape((len(years),8746))
        for i in range(len(years)):
            Y2[i,:] = Y2[i,:] - Y[i,:]

        for i in range(1,len(years)):
           Y2[i,:] = Y2[i,:] + Y2[i-1,-1]
        Y2 = Y2.flatten()
        X2 = np.asarray(X2).flatten()
        #print(np.shape(Y2))
        #X = np.asarray(X2).reshape((8746,18))
        plt.plot(X2,Y2)
    # plt.twinx()
    # for location in locations:
    #     Y = []
    #     for year in years:
    #         TempX,TempY = fetch('PERC_'+location+'_'+str(year),'TAU550')
    #         TempY = TempY[:8746]
    #         Y.append(TempY)
    #     Y = np.asarray(Y).flatten()
    #     N = int(8764)
    #     Y = np.convolve(Y,np.ones((N,))/N, mode='valid')
    #     #Y = np.asarray(Y).reshape((113698,))
    #     #Y = np.asarray(Y).flatten()
    #     #x = [datetime.datetime(year=year,month=12,day=31) for year in years]
    #     plt.plot(X2[int(8764)-1:],Y,linestyle='--')
    plt.show()

def draw_CISO():
    data_dir = os.path.join(os.getcwd(),'CISO-data')
    data_dirs = os.listdir(data_dir)
    data_dirs = [os.path.join(data_dir,dir) for dir in data_dirs if '.xml.zip' not in dir]
    data_dirs = [pd.read_csv(dir,compression='zip') for dir in data_dirs]
    data = pd.concat(data_dirs)

    data = data[data['RENEWABLE_TYPE'] == 'Solar']


    #data = data.drop(columns=['LABEL','XML_DATA_ITEM','MARKET_RUN_ID_POS','RENEW_POS','MARKET_RUN_ID','GROUP','RENEWABLE_TYPE','INTERVALENDTIME_GMT','OPR_INTERVAL'])
    data = data.drop(columns=['LABEL','XML_DATA_ITEM','MARKET_RUN_ID_POS','RENEW_POS','MARKET_RUN_ID','GROUP','RENEWABLE_TYPE','INTERVALSTARTTIME_GMT','INTERVALENDTIME_GMT','OPR_INTERVAL'])

    data['OPR_HR'].values[data['OPR_HR'] >= 24] = 0
    Date = data['OPR_DT'] +' '+ data['OPR_HR'].astype(str)+':00:00'
    #Date = data['INTERVALSTARTTIME_GMT']
    data = data.set_index(pd.DatetimeIndex(Date))
    data = data.drop(columns=['OPR_DT','OPR_HR'])

    d = data.sort_index()#[data['TRADING_HUB'] == 'NP15']
    d = d.resample('3H').mean()

    b = data.sort_index()
    b = b[b != 0 ]
    print(data.index)
    b = b.rolling(24*15,min_periods=1).mean().resample('D').mean()

    plt.plot(d)
    plt.plot(b,linestyle='--')
    return d.index

def draw_vlines(Dates, method='repeat_yearly', year=2020, month=1, day=1, hour=0, ymin=-500, ymax=500):
    match method:
        case 'repeat_yearly':
            Dates = pd.DatetimeIndex(Dates)
            start_year = Dates.year[0]
            end_year = Dates.year[-1]
            selected_dates = [str(y)+'-'+str(month)+'-'+str(day) for y in np.arange(start_year,end_year+1)]
            selected_dates = pd.to_datetime(selected_dates)
            Dates = pd.DataFrame(index=Dates)
            Dates_idx = Dates[Dates.index.normalize().isin(selected_dates)][::8]
            plt.vlines(Dates_idx.index.values,ymin=ymin,ymax=ymax,colors='tab:red',linestyles='--')
        case 'single':
            Dates = pd.DatetimeIndex(Dates)
            start_year = Dates.year[0]
            end_year = Dates.year[-1]
            selected_dates = [str(year)+'-'+str(month)+'-'+str(day)]
            selected_dates = pd.to_datetime(selected_dates)
            Dates = pd.DataFrame(index=Dates)
            Dates_idx = Dates[Dates.index.normalize().isin(selected_dates)][::8]
            plt.vlines(Dates_idx.index.values,ymin=ymin,ymax=ymax,colors='tab:red',linestyles='--')
    return  

def plot_from_file(dir,start_year,end_year,num_average_years,colour=''):
    dir = os.path.join(os.getcwd(),dir+'.csv')
    data = pd.read_csv(dir)
    names = []
    for i in range(len(data)):
        name = str(data.loc[i]['Name']) + '_' + str(data.loc[i]['State'])
        names.append(name)
    plot_multiyear(names,start_year,end_year,num_average_years,colour=colour)
    return

def calc_from_file(dir,start_year,end_year):
    dir = os.path.join(os.getcwd(),dir+'.csv')
    data = pd.read_csv(dir)
    names = []
    for i in range(len(data)):
        name = str(data.loc[i]['Name']) + '_' + str(data.loc[i]['State'])
        names.append(name)
    dates,Y = calc_multiyear(names,start_year,end_year)
    return Y

def plot_from_file_spacial(dir,start_year,end_year):
    dir = os.path.join(os.getcwd(),dir+'.csv')
    data = pd.read_csv(dir)
    names = []
    for i in range(len(data)):
        name = str(data.loc[i]['Name']) + '_' + str(data.loc[i]['State'])
        names.append(name)
    dates,data_z = calc_multiyear(names,start_year,end_year)
    z = np.zeros((14,14,2920))
    unique_lat = data['Latitude'].unique()
    unique_lon = data['Longitude'].unique()

    for idx,lat in enumerate(unique_lat):
        for jdx,lon in enumerate(unique_lon):
            index = data.index[(data['Latitude'] == lat)&(data['Longitude'] == lon)].values[0]
            z[jdx,idx,:] = data_z[index]
    d = dates.values
    d_filter = dates[(dates.hour == 12) & (((dates.day >= 14) & (dates.month == 7)) | ((dates.day <= 10) & (dates.month == 8)))].values
    d = np.searchsorted(d,d_filter)
    plt.rcParams["figure.figsize"] = (11,7)
    fig, ax = plt.subplots(ncols=7,nrows=4)
    x = 0
    y = 0
    for i in d:
        im = ax[y,x].imshow(z[:,:,i], cmap='inferno_r',vmin=-100,vmax=0)
        ax[y,x].set_axis_off()
        ax[y,x].title.set_text(str(dates[i].day)+'-'+str(dates[i].month)+'-'+str(dates[i].year))
        x += 1
        if x >= 7:
            x = 0
            y += 1
    cbar_ax = fig.add_axes([0.905, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax,label='Energy Lost (Wm$^{-2}$)')
    dir = os.path.join(os.getcwd(),'Figures','California_2020_Spectral_NO_Fire')
    plt.savefig(dir+'.png',dpi=1200)
    plt.savefig(dir+'.svg')
    return

#plot_from_file('WestOfRockies_coast',2008,2016,5,colour='tab:blue')
#plot_from_file('WestOfRockies_inland',2008,2016,5,colour='tab:orange')

def California_2020():
    Dates = plot_multiyear(['Malibu', 'Santa_Clarita', 'Santa_Rosa'],2020,2020,5)
    draw_vlines(Dates,month=8,day=16,ymin=-30,ymax=0)
    plt.ylabel('Energy Lost (W/m$^{-2}$)')
    plt.ylim(top=0,bottom=-30)
    plt.tight_layout()
    dir = os.path.join(os.getcwd(),'Figures','California_2020')
    plt.savefig(dir+'.png',dpi=1200)
    plt.savefig(dir+'.svg')

def California_2020_CISO():
    Dates = draw_CISO()
    draw_vlines(Dates,month=8,day=16,hour=12,ymin=0,ymax=4000)
    plt.ylim(top=4000,bottom=0)
    plt.ylabel('Energy Generated(MW)')
    plt.tight_layout()
    dir = os.path.join(os.getcwd(),'Figures','California_CISO_2020')
    plt.savefig(dir+'.png',dpi=1200)
    plt.savefig(dir+'.svg')

def California_2008_2020():
    Dates = plot_multiyear(['Malibu', 'Santa_Clarita', 'Santa_Rosa'],2008,2020,1)
    draw_vlines(Dates,method='single',year=2020,month=8,day=16,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2019,month=10,day=23,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2018,month=7,day=27,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2017,month=12,day=4,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2016,month=7,day=22,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2015,month=7,day=31,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2014,month=8,day=14,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2013,month=8,day=17,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2012,month=8,day=12,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2011,month=9,day=10,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2010,month=7,day=27,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2009,month=8,day=26,ymin=-30,ymax=0)
    draw_vlines(Dates,method='single',year=2008,month=6,day=21,ymin=-30,ymax=0)
    plt.ylabel('Energy Lost (W/m$^{-2}$)')
    plt.ylim(top=0,bottom=-30)
    plt.tight_layout()
    dir = os.path.join(os.getcwd(),'Figures','California_2008_2020')
    #plt.show()
    plt.savefig(dir+'.png',dpi=1200)
    plt.savefig(dir+'.svg')


def San_Juan_2020():
    Dates = plot_multiyear(['San_Juan'],2020,2020,1)
    draw_vlines(Dates,month=6,day=23,ymin=-25,ymax=-5)
    dir = os.path.join(os.getcwd(),'Figures','San_Juan_2020')
    #plt.show()
    plt.ylim(top=-5,bottom=-20)
    plt.savefig(dir+'.png',dpi=1200)
    plt.savefig(dir+'.svg')


def San_Juan_2010_2020():
    Dates = plot_multiyear(['San_Juan'],2003,2020,1)
    draw_vlines(Dates,month=6,day=23,ymin=-25,ymax=-5)
    dir = os.path.join(os.getcwd(),'Figures','San_Juan_2003_2020')
    #plt.show()
    plt.ylim(top=-5,bottom=-20)
    plt.savefig(dir+'.png',dpi=1200)
    plt.savefig(dir+'.svg')


def States_Political():
    plot_from_file('Republican',2003,2016,1,'tab:red')
    plot_from_file('Democrat',2003,2016,1,'tab:blue') 
    plt.ylabel('Energy Lost (Wm$^{-2}$)')
    dir = os.path.join(os.getcwd(),'Figures','States_Political_2020')
    #plt.show()
    plt.savefig(dir+'.png',dpi=1200)
    plt.savefig(dir+'.svg') 

def States_Scatter():
    dir = 'PopulusAmerica'
    for i in range(2016,2017):
        dir = 'PopulusAmerica'
        y = calc_from_file(dir,i,i)
        data = np.mean(y,axis=1)
        dir = os.path.join(os.getcwd(),dir+'.csv')
        xs = pd.read_csv(dir)
        x = xs['Population Density'].values.ravel()
        c = xs['Latitude'].values.ravel()
        l = xs['State'].values.ravel()
        for i in range(len(x)):
            plt.scatter(x[i]*0.386102158542446,data[i],label=l[i])
    plt.legend()
    plt.ylabel('Mean Energy Lost (Wm$^{-2}$)')
    plt.xlabel('Polulation Per Km (people Km$^{-1}$)')
    

#California_2020_CISO()
#plt.clf()
#California_2020()
#California_2008_2020()
#plot_from_file_spacial('California',2020,2020,)

States_Scatter()
#plot_from_file('Republican',2003,2016,1,'tab:red')
#plot_from_file('Democrat',2003,2016,1,'tab:blue')
#San_Juan_2020()
plt.show()