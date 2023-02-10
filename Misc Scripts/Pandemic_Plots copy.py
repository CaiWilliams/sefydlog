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
    xvals = np.arange(len(data)*3 -2)/3
    yvals = np.interp(xvals,x,data.to_numpy())
    xdates = pd.date_range(Data['Date'].loc[0],Data['Date'].loc[len(data)-1],freq='h')
    return xdates,yvals

def calc_power_diff(A, B, linestyle="-", colour='tab:blue'):
    fileA = os.path.join(os.getcwd(),'Results',A+'.csv')
    fileB = os.path.join(os.getcwd(),'Results',B+'.csv')
    dA = pd.read_csv(fileA)
    dB = pd.read_csv(fileB)
    data = np.cumsum(dB['Pmax'] - dA['Pmax'])
    x = np.arange(len(data))
    xvals = np.arange(len(data)*3 -2)/3
    yvals = np.interp(xvals,x,data.to_numpy())
    xdates = pd.date_range(dA['Date'].loc[0],dA['Date'].loc[len(dA)-1],freq='h')
    return xdates,yvals

def covid_California():
    x11,y11 = calc_power_diff('PERC_Malibu_pristine_2020','PERC_Malibu_2020',colour='tab:blue')
    x12,y12 = calc_power_diff('PERC_Santa_Rosa_pristine_2020','PERC_Santa_Rosa_2020',colour='tab:orange')
    x13,y13 = calc_power_diff('PERC_Santa_Clarita_pristine_2020','PERC_Santa_Clarita_2020',colour='tab:green')

    x21,y21 = calc_power_diff('PERC_Malibu_pristine_2019','PERC_Malibu_2019',colour='tab:blue',linestyle="--")
    x22,y22 = calc_power_diff('PERC_Santa_Rosa_pristine_2019','PERC_Santa_Rosa_2019',colour='tab:orange',linestyle="--")
    x23,y23 = calc_power_diff('PERC_Santa_Clarita_pristine_2019','PERC_Santa_Clarita_2019',colour='tab:green',linestyle="--")

    plt.plot(x22[:8746],y11[:8746]-y21[:8746])
    plt.plot(x22[:8746],y12[:8746]-y22[:8746])
    plt.plot(x22[:8746],y13[:8746]-y23[:8746])
    plt.vlines(78,np.min(y12[:8746]-y22[:8746]),np.max(y13[:8746]-y23[:8746]),color='tab:red',linestyles='--')
    plt.vlines(230,np.min(y12[:8746]-y22[:8746]),np.max(y13[:8746]-y23[:8746]),color='tab:purple',linestyles='--')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    plt.tight_layout()
    #calc_power_diff('PERC_Santa_Rosa_pristine_2019','PERC_Santa_Rosa_2019',colour='tab:orange',linestyle="--")
    #calc_power_diff('PERC_Santa_Clarita_pristine_2019','PERC_Santa_Clarita_2019',colour='tab:green',linestyle="--")
    return

def covid_California_2():
    locations = ['Malibu', 'Santa_Rosa','Santa_Clarita']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y
        X,Y2 = calc_power_diff('PERC_'+location+'_pristine_2020','PERC_'+location+'_2020')
        Y = Y/len(years)
        Y = Y2[:8746] - Y
        X = X[:8746]
        plt.plot(X,Y)
    plt.show()
    return

def covid_California_3():
    locations = ['Malibu','Santa_Rosa','Santa_Clarita']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y

        Y = np.asarray(Y).reshape(1,8746)
        X2 = []
        Y2 = []
        years = np.arange(2008,2021)
        for year in years:
            Temp_X2,Temp_Y2 = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Temp_X2 = Temp_X2[:8746]
            Temp_Y2 = Temp_Y2[:8746]
            X2.append(Temp_X2)
            Y2.append(Temp_Y2)
        
        Y = Y/len(years)
        Y2 = np.asarray(Y2).reshape((len(years),8746))
        for i in range(len(years)):
            Y2[i,:] = Y2[i,:] - Y[:]

        for i in range(1,len(years)):
           Y2[i,:] = Y2[i,:] + Y2[i-1,-1]
        Y2 = Y2.flatten()
        X2 = np.asarray(X2).flatten()
        #X = np.asarray(X2).reshape((8746,18))
        plt.plot(X2,Y2)
    plt.show()
    return

def covid_California_4():
    locations = ['Malibu','Santa_Rosa','Santa_Clarita']
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
    #plt.twinx()
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
    return

def covid_Mexico_City():
    x1,y1 = calc_power_diff('PERC_Mexico_City_pristine_2020','PERC_Mexico_City_2020')
    x2,y2 = calc_power_diff('PERC_Mexico_City_pristine_2019','PERC_Mexico_City_2019')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])
    plt.vlines(152,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    #plt.vlines(197,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:orange')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    
def covid_Mexico_City_2():
    locations = ['Mexico_City']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y
        X,Y2 = calc_power_diff('PERC_'+location+'_pristine_2020','PERC_'+location+'_2020')
        Y = Y/len(years)
        Y = Y2[:8746] - Y
        X = X[:8746]
        plt.plot(X,Y)
    plt.show()
    return

def covid_Mexico_City_3():
    locations = ['Mexico_City']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y

        Y = np.asarray(Y).reshape(1,8746)
        X2 = []
        Y2 = []

        for year in years:
            Temp_X2,Temp_Y2 = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Temp_X2 = Temp_X2[:8746]
            Temp_Y2 = Temp_Y2[:8746]
            X2.append(Temp_X2)
            Y2.append(Temp_Y2)
        
        Y = Y/len(years)
        Y2 = np.asarray(Y2).reshape((len(years),8746))
        for i in range(len(years)):
            Y2[i,:] = Y2[i,:] - Y[:]

        for i in range(1,len(years)):
           Y2[i,:] = Y2[i,:] + Y2[i-1,-1]
        Y2 = Y2.flatten()
        X2 = np.asarray(X2).flatten()
        #X = np.asarray(X2).reshape((8746,18))
        plt.plot(X2,Y2)
    plt.show()
    return

def covid_Mexico_City_4():
    locations = ['Mexico_City']
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
    return

def covid_Durham():
    x1,y1 = calc_power_diff('PERC_Durham_pristine_2020','PERC_Durham_2020')
    x2,y2 = calc_power_diff('PERC_Durham_pristine_2019','PERC_Durham_2019')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])
    plt.vlines(82,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    #plt.vlines(197,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:orange')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def covid_Durham_2():
    locations = ['Durham']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y
        X,Y2 = calc_power_diff('PERC_'+location+'_pristine_2020','PERC_'+location+'_2020')
        Y = Y/len(years)
        Y = Y2[:8746] - Y
        X = X[:8746]
        plt.plot(X,Y)
    plt.show()
    return

def covid_Durham_3():
    locations = ['Durham']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y

        Y = np.asarray(Y).reshape(1,8746)
        X2 = []
        Y2 = []

        for year in years:
            Temp_X2,Temp_Y2 = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Temp_X2 = Temp_X2[:8746]
            Temp_Y2 = Temp_Y2[:8746]
            X2.append(Temp_X2)
            Y2.append(Temp_Y2)
        
        Y = Y/len(years)
        Y2 = np.asarray(Y2).reshape((len(years),8746))
        for i in range(len(years)):
            Y2[i,:] = Y2[i,:] - Y[:]

        for i in range(1,len(years)):
           Y2[i,:] = Y2[i,:] + Y2[i-1,-1]
        Y2 = Y2.flatten()
        X2 = np.asarray(X2).flatten()
        #X = np.asarray(X2).reshape((8746,18))
        plt.plot(X2,Y2)
    plt.show()
    return

def covid_Durham_4():
    locations = ['Durham']
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
    return

def covid_Lahore():
    x1,y1 = calc_power_diff('PERC_Lahore_pristine_2020','PERC_Lahore_2020')
    x2,y2 = calc_power_diff('PERC_Lahore_pristine_2019','PERC_Lahore_2019')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])
    plt.vlines(82,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def covid_Lahore_3():
    locations = ['Lahore']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y

        Y = np.asarray(Y).reshape(1,8746)
        X2 = []
        Y2 = []

        for year in years:
            Temp_X2,Temp_Y2 = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Temp_X2 = Temp_X2[:8746]
            Temp_Y2 = Temp_Y2[:8746]
            X2.append(Temp_X2)
            Y2.append(Temp_Y2)
        
        Y = Y/len(years)
        Y2 = np.asarray(Y2).reshape((len(years),8746))
        for i in range(len(years)):
            Y2[i,:] = Y2[i,:] - Y[:]

        for i in range(1,len(years)):
           Y2[i,:] = Y2[i,:] + Y2[i-1,-1]
        Y2 = Y2.flatten()
        X2 = np.asarray(X2).flatten()
        #X = np.asarray(X2).reshape((8746,18))
        plt.plot(X2,Y2)
    plt.show()
    return

def covid_Lahore_4():
    locations = ['Lahore']
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
    return

def covid_San_Juan():
    x1,y1 = calc_power_diff('PERC_San_Juan_pristine_2020','PERC_San_Juan_2020')
    x2,y2 = calc_power_diff('PERC_San_Juan_pristine_2019','PERC_San_Juan_2019')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])
    plt.vlines(73,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    plt.vlines(175,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:orange',linestyles='--')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def covid_San_Juan_3():
    locations = ['San_Juan']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y

        Y = np.asarray(Y).reshape(1,8746)
        X2 = []
        Y2 = []

        for year in years:
            Temp_X2,Temp_Y2 = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Temp_X2 = Temp_X2[:8746]
            Temp_Y2 = Temp_Y2[:8746]
            X2.append(Temp_X2)
            Y2.append(Temp_Y2)
        
        Y = Y/len(years)
        Y2 = np.asarray(Y2).reshape((len(years),8746))
        for i in range(len(years)):
            Y2[i,:] = Y2[i,:] - Y[:]

        for i in range(1,len(years)):
           Y2[i,:] = Y2[i,:] + Y2[i-1,-1]
        Y2 = Y2.flatten()
        X2 = np.asarray(X2).flatten()
        #X = np.asarray(X2).reshape((8746,18))
        plt.plot(X2,Y2)
    plt.show()

def covid_San_Juan_4():
    locations = ['San_Juan']
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
    return

def covid_Zhengzhou():
    x1,y1 = calc_power_diff('PERC_Zhengzhou_pristine_2020','PERC_Zhengzhou_2020')
    x2,y2 = calc_power_diff('PERC_Zhengzhou_pristine_2019','PERC_Zhengzhou_2019')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def covid_Zhengzhou_3():
    locations = ['Zhengzhou']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y

        Y = np.asarray(Y).reshape(1,8746)
        X2 = []
        Y2 = []

        for year in years:
            Temp_X2,Temp_Y2 = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Temp_X2 = Temp_X2[:8746]
            Temp_Y2 = Temp_Y2[:8746]
            X2.append(Temp_X2)
            Y2.append(Temp_Y2)
        
        Y = Y/len(years)
        Y2 = np.asarray(Y2).reshape((len(years),8746))
        for i in range(len(years)):
            Y2[i,:] = Y2[i,:] - Y[:]

        for i in range(1,len(years)):
           Y2[i,:] = Y2[i,:] + Y2[i-1,-1]
        Y2 = Y2.flatten()
        X2 = np.asarray(X2).flatten()
        #X = np.asarray(X2).reshape((8746,18))
        plt.plot(X2,Y2)
    plt.show()

def covid_Zhengzhou_4():
    locations = ['Zhengzhou']
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
    return

def Olympics():
    x1,y1 = calc_power_diff('PERC_Beijing_pristine_2008','PERC_Beijing_2008')
    x2,y2 = calc_power_diff('PERC_Beijing_pristine_2007','PERC_Beijing_2007')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])

    x1,y1 = calc_power_diff('PERC_Chengdu_pristine_2008','PERC_Chengdu_2008')
    x2,y2 = calc_power_diff('PERC_Chengdu_pristine_2007','PERC_Chengdu_2007')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])

    x1,y1 = calc_power_diff('PERC_Guangzhou_pristine_2008','PERC_Guangzhou_2008')
    x2,y2 = calc_power_diff('PERC_Guangzhou_pristine_2007','PERC_Guangzhou_2007')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])

    plt.vlines(220,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    plt.vlines(236,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def Olympics_3():
    locations = ['Beijing','Chengdu','Guangzhou']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y

        Y = np.asarray(Y).reshape(1,8746)
        X2 = []
        Y2 = []

        for year in years:
            Temp_X2,Temp_Y2 = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Temp_X2 = Temp_X2[:8746]
            Temp_Y2 = Temp_Y2[:8746]
            X2.append(Temp_X2)
            Y2.append(Temp_Y2)
        
        Y = Y/len(years)
        Y2 = np.asarray(Y2).reshape((len(years),8746))
        for i in range(len(years)):
            Y2[i,:] = Y2[i,:] - Y[:]

        for i in range(1,len(years)):
           Y2[i,:] = Y2[i,:] + Y2[i-1,-1]
        Y2 = Y2.flatten()
        X2 = np.asarray(X2).flatten()
        #X = np.asarray(X2).reshape((8746,18))
        plt.plot(X2,Y2)
    plt.show()

def Olympics_4():
    locations = ['Beijing','Chengdu','Guangzhou']
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
    return

def Olympics_Cities_3():
    locations = ['Beijing','Rio','London','Tokyo']
    for location in locations:
        years = np.arange(2003,2021)
        Y = np.zeros(8746)
        for year in years:
            X,Temp_Y = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Y = Temp_Y[:8746] + Y

        Y = np.asarray(Y).reshape(1,8746)
        X2 = []
        Y2 = []

        for year in years:
            Temp_X2,Temp_Y2 = calc_power_diff('PERC_'+location+'_pristine_'+str(year),'PERC_'+location+'_'+str(year))
            Temp_X2 = Temp_X2[:8746]
            Temp_Y2 = Temp_Y2[:8746]
            X2.append(Temp_X2)
            Y2.append(Temp_Y2)
        
        Y = Y/len(years)
        Y2 = np.asarray(Y2).reshape((len(years),8746))
        for i in range(len(years)):
            Y2[i,:] = Y2[i,:] - Y[:]

        for i in range(1,len(years)):
           Y2[i,:] = Y2[i,:] + Y2[i-1,-1]
        Y2 = Y2.flatten()
        X2 = np.asarray(X2).flatten()
        #X = np.asarray(X2).reshape((8746,18))
        plt.plot(X2,Y2)
    plt.show()

def Olympics_Cities_4():
    locations = ['Beijing','Rio','London','Tokyo']
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

def Volcano_Durham():
    x1,y1 = calc_power_diff('PERC_Durham_pristine_2010','PERC_Durham_2010')
    x2,y2 = calc_power_diff('PERC_Durham_pristine_2009','PERC_Durham_2009')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])

    x1,y1 = calc_power_diff('PERC_Montreal_pristine_2010','PERC_Montreal_2010')
    x2,y2 = calc_power_diff('PERC_Montreal_pristine_2009','PERC_Montreal_2009')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])


    
    plt.vlines(79,-200,300,color='tab:red',linestyles='--')
    plt.vlines(174,-200,300,color='tab:red',linestyles='--')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def Volcano_Durham_4():
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

Volcano_Durham_4()
#covid_Zhengzhou_4()
#plot('PERC_Mexico_City_SDM_2019')
#covid_Durham_2()
# plt.savefig('Fig2_Covid_Mexico_City.png',dpi=600)
# plt.clf()
# covid_San_Juan()
# plt.savefig('Fig2_Covid_San_Juan.png',dpi=600)
# plt.clf()
# covid_Lahore()
# plt.savefig('Fig2_Covid_Lahore.png',dpi=600)
# plt.clf()
# covid_Durham()
# plt.savefig('Fig2_Covid_Durham.png',dpi=600)
# plt.clf()
# covid_Zhengzhou()
# plt.savefig('Fig2_Covid_Zhengzhou.png',dpi=600)
# plt.clf()
# Volcano_Durham()
# plt.savefig('Fig2_Volcano.png',dpi=600)
# plt.clf()
# Olympics()
# plt.savefig('Fig2_Olympics.png',dpi=600)
# plt.clf()
# covid_California()
# plt.savefig('Fig2_Covid_California.png',dpi=600)
# plt.clf()
# #plt.show()