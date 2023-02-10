import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot(name):
    file = os.path.join(os.getcwd(),'Results',name+'.csv')
    data = pd.read_csv(file)
    data = data['Pmax']
    plt.plot(np.cumsum(data[::]))
    return data

def calc_power_diff(A, B, linestyle="-", colour='tab:blue'):
    fileA = os.path.join(os.getcwd(),'Results',A+'.csv')
    fileB = os.path.join(os.getcwd(),'Results',B+'.csv')
    dA = pd.read_csv(fileA)
    dB = pd.read_csv(fileB)
    data = np.cumsum(dB['Pmax'] - dA['Pmax'])
    x = np.arange(len(data))
    xvals = np.arange(len(data)*3 -2)/3
    yvals = np.interp(xvals,x,data.to_numpy())
    xdates = pd.date_range(dA['Date'].loc[0],dA['Date'].loc[len(dA)-1],freq='h').day_of_year
    return xdates,yvals

def covid_California():
    x11,y11 = calc_power_diff('P3HTPCBM_Malibu_Pristine_2020','P3HTPCBM_Malibu_2020',colour='tab:blue')
    x12,y12 = calc_power_diff('P3HTPCBM_Santa_Rosa_Pristine_2020','P3HTPCBM_Santa_Rosa_2020',colour='tab:orange')
    x13,y13 = calc_power_diff('P3HTPCBM_Santa_Clarita_Pristine_2020','P3HTPCBM_Santa_Clarita_2020',colour='tab:green')

    x21,y21 = calc_power_diff('P3HTPCBM_Malibu_Pristine_2019','P3HTPCBM_Malibu_2019',colour='tab:blue',linestyle="--")
    x22,y22 = calc_power_diff('P3HTPCBM_Santa_Rosa_Pristine_2019','P3HTPCBM_Santa_Rosa_2019',colour='tab:orange',linestyle="--")
    x23,y23 = calc_power_diff('P3HTPCBM_Santa_Clarita_Pristine_2019','P3HTPCBM_Santa_Clarita_2019',colour='tab:green',linestyle="--")

    plt.plot(x22,y11[:8746]-y21[:8746])
    plt.plot(x22,y12[:8746]-y22[:8746])
    plt.plot(x22,y13[:8746]-y23[:8746])
    plt.vlines(78,np.min(y12[:8746]-y22[:8746]),np.max(y13[:8746]-y23[:8746]),color='tab:red',linestyles='--')
    plt.vlines(230,np.min(y12[:8746]-y22[:8746]),np.max(y13[:8746]-y23[:8746]),color='tab:purple',linestyles='--')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    plt.tight_layout()
    #calc_power_diff('P3HTPCBM_Santa_Rosa_Pristine_2019','P3HTPCBM_Santa_Rosa_2019',colour='tab:orange',linestyle="--")
    #calc_power_diff('P3HTPCBM_Santa_Clarita_Pristine_2019','P3HTPCBM_Santa_Clarita_2019',colour='tab:green',linestyle="--")
    return

def covid_Mexico_City():
    x1,y1 = calc_power_diff('P3HTPCBM_Mexico_City_Pristine_2020','P3HTPCBM_Mexico_City_2020')
    x2,y2 = calc_power_diff('P3HTPCBM_Mexico_City_Pristine_2019','P3HTPCBM_Mexico_City_2019')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])
    plt.vlines(152,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    #plt.vlines(197,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:orange')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def covid_Durham():
    x1,y1 = calc_power_diff('PERC_Durham_pristine_2004','PERC_Durham_2004')
    x2,y2 = calc_power_diff('PERC_Durham_pristine_2003','PERC_Durham_2003')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])
    plt.vlines(82,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    #plt.vlines(197,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:orange')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def covid_Lahore():
    x1,y1 = calc_power_diff('P3HTPCBM_Lahore_Pristine_2020','P3HTPCBM_Lahore_2020')
    x2,y2 = calc_power_diff('P3HTPCBM_Lahore_Pristine_2019','P3HTPCBM_Lahore_2019')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])
    plt.vlines(82,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def covid_San_Juan():
    x1,y1 = calc_power_diff('P3HTPCBM_San_Juan_Pristine_2020','P3HTPCBM_San_Juan_2020')
    x2,y2 = calc_power_diff('P3HTPCBM_San_Juan_Pristine_2019','P3HTPCBM_San_Juan_2019')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])
    plt.vlines(73,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    plt.vlines(175,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:orange',linestyles='--')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def covid_Zhengzhou():
    x1,y1 = calc_power_diff('P3HTPCBM_Zhengzhou_Pristine_2020','P3HTPCBM_Zhengzhou_2020')
    x2,y2 = calc_power_diff('P3HTPCBM_Zhengzhou_Pristine_2019','P3HTPCBM_Zhengzhou_2019')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def Olympics():
    x1,y1 = calc_power_diff('P3HTPCBM_Beijing_Pristine_2008','P3HTPCBM_Beijing_2008')
    x2,y2 = calc_power_diff('P3HTPCBM_Beijing_Pristine_2007','P3HTPCBM_Beijing_2007')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])

    x1,y1 = calc_power_diff('P3HTPCBM_Chengdu_Pristine_2008','P3HTPCBM_Chengdu_2008')
    x2,y2 = calc_power_diff('P3HTPCBM_Chengdu_Pristine_2007','P3HTPCBM_Chengdu_2007')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])

    x1,y1 = calc_power_diff('P3HTPCBM_Guangzhou_Pristine_2008','P3HTPCBM_Guangzhou_2008')
    x2,y2 = calc_power_diff('P3HTPCBM_Guangzhou_Pristine_2007','P3HTPCBM_Guangzhou_2007')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])

    plt.vlines(220,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    plt.vlines(236,np.min(y1[:8746]-y2[:8746]),np.max(y1[:8746]-y2[:8746]),color='tab:red',linestyles='--')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return

def Volcano_Durham():
    x1,y1 = calc_power_diff('P3HTPCBM_Durham_Pristine_2010','P3HTPCBM_Durham_2010')
    x2,y2 = calc_power_diff('P3HTPCBM_Durham_Pristine_2009','P3HTPCBM_Durham_2009')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])

    x1,y1 = calc_power_diff('P3HTPCBM_Montreal_Pristine_2010','P3HTPCBM_Montreal_2010')
    x2,y2 = calc_power_diff('P3HTPCBM_Montreal_Pristine_2009','P3HTPCBM_Montreal_2009')
    plt.plot(x2[:8746],y1[:8746]-y2[:8746])


    
    plt.vlines(79,-200,300,color='tab:red',linestyles='--')
    plt.vlines(174,-200,300,color='tab:red',linestyles='--')
    plt.xlabel('Day of Year')
    plt.ylabel('Difference in Human Impact on Solar Yeild (Whm$^{-2}$)')
    return
plot('PERC_Durham_2004')
plot('PERC_Durham_pristine_2004')
plt.show()
#
#plot('P3HTPCBM_Mexico_City_SDM_2019')
# covid_Mexico_City()
# plt.savefig('Fig_Covid_Mexico_City.png',dpi=600)
# plt.clf()
# covid_San_Juan()
# plt.savefig('Fig_Covid_San_Juan.png',dpi=600)
# plt.clf()
# covid_Lahore()
# plt.savefig('Fig_Covid_Lahore.png',dpi=600)
# plt.clf()
# covid_Durham()
# plt.savefig('Fig_Covid_Durham.png',dpi=600)
# plt.clf()
# covid_Zhengzhou()
# plt.savefig('Fig_Covid_Zhengzhou.png',dpi=600)
# plt.clf()
# Volcano_Durham()
# plt.savefig('Fig_Volcano.png',dpi=600)
# plt.clf()
# Olympics()
# plt.savefig('Fig_Olympics.png',dpi=600)
# plt.clf()
# covid_California()
# plt.savefig('Fig_Covid_California.png',dpi=600)
# plt.clf()
plt.show()