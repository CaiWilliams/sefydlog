import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot(name):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    data = pd.read_csv(file)
    data = data['PCE']
    plt.plot(data[::24])
    return data

def plot_diff(A,B):
    fileA = os.path.join(os.getcwd(),'Results',A+'.csv')
    fileB = os.path.join(os.getcwd(),'Results',B+'.csv')
    dA = pd.read_csv(fileA)
    dB = pd.read_csv(fileB)
    data = ((dB['Pmax'] - dA['Pmax']) / dA['Pmax'])
    plt.plot(dB['Date'][::24], data[::24])
    plt.xticks(np.linspace(0,len(dB['Date'][::24])-4,3))
    plt.ylabel('Change in PCE due to Polution (%)')
    return data

def calc_power(name):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    data = pd.read_csv(file)
    data = data['Pmax']
    x = np.arange(len(data))
    xvals = np.arange(len(data)*3)/3
    yvals = np.interp(xvals,x,data.to_numpy())
    plt.plot(np.cumsum(yvals)/1e3)
    return

def calc_power_diff(A, B):
    fileA = os.path.join(os.getcwd(),'Results',A+'.csv')
    fileB = os.path.join(os.getcwd(),'Results',B+'.csv')
    dA = pd.read_csv(fileA)
    dB = pd.read_csv(fileB)
    data = np.cumsum(dB['Pmax'] - dA['Pmax'])
    x = np.arange(len(data))
    xvals = np.arange(len(data)*3)/3
    yvals = np.interp(xvals,x,data.to_numpy())
    plt.plot(yvals)
    plt.xticks(np.linspace(0,len(xvals),13),labels=[1,2,3,4,5,6,7,8,9,10,11,12,1])
    plt.ylabel('Diffrence in Generation (Wh * m$^{-2}$)')
    return



# plot_diff('P3HTPCBM_Mexico_City_Pristine_2019','P3HTPCBM_Mexico_City_2019')
# plot_diff('P3HTPCBM_Zhengzhou_Pristine_2019','P3HTPCBM_Zhengzhou_2019')
# plot_diff('P3HTPCBM_Lahore_Pristine_2019','P3HTPCBM_Lahore_2019')
# plot_diff('P3HTPCBM_San_Juan_Pristine_2019','P3HTPCBM_San_Juan_2019')
# plot_diff('P3HTPCBM_Durham_Pristine_2019','P3HTPCBM_Durham_2019')

# plot('P3HTPCBM_Mexico_City_2019')
# plot('P3HTPCBM_Zhengzhou_2019')
# plot('P3HTPCBM_Lahore_2019')
# plot('P3HTPCBM_Lahore_Pristine_2019')
# plot('P3HTPCBM_San_Juan_2019')
# plot('P3HTPCBM_Durham_2019')

#calc_power('P3HTPCBM_Malibu_2019')
calc_power('P3HTPCBM_Santa_Rosa_2019')
calc_power('P3HTPCBM_Santa_Rosa_2020')
#calc_power('P3HTPCBM_Santa_Clarita_2019')

#calc_power('P3HTPCBM_Beijing_2008')
#calc_power('P3HTPCBM_Beijing_Pristine_2008')

#calc_power('P3HTPCBM_London_2008')
#calc_power('P3HTPCBM_London_Pristine_2008')

#calc_power('P3HTPCBM_Chengdu_2008')
#calc_power('P3HTPCBM_Guangzhou_2008')
#calc_power('P3HTPCBM_Zheng_2008')

#calc_power_diff('P3HTPCBM_Santa_Rosa_Pristine_2019','P3HTPCBM_Santa_Rosa_2019')
#calc_power_diff('P3HTPCBM_Santa_Rosa_Pristine_2020','P3HTPCBM_Santa_Rosa_2020')
#calc_power_diff('P3HTPCBM_Guangzhou_Pristine_2008','P3HTPCBM_Guangzhou_2008')

#plot_diff('P3HTPCBM_Beijing_Pristine_2008','P3HTPCBM_Beijing_2008')
#plot_diff('P3HTPCBM_London_Pristine_2008','P3HTPCBM_London_2008')
#plot_diff('P3HTPCBM_Rio_Pristine_2008','P3HTPCBM_Rio_2008')

#plot_diff('P3HTPCBM_Chengdu_Pristine_2008','P3HTPCBM_Chengdu_2008')

plt.show()