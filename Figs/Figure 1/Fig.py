import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'Location Lists','Californian_Fires.csv'))
x = data['Year'].to_numpy()
y1 = data['Fires'].to_numpy()
y2 = data['Acers'].to_numpy()/247.1

z1 = np.polyfit(x,y1,2)
p1 = np.poly1d(z1)

z2 = np.polyfit(x,y2,2)
p2 = np.poly1d(z2)

xp = np.linspace(x[0]-1,x[-1]+1,10000)

idx2020 = np.argwhere(x == 2020)

plt.scatter(x,y1,color='tab:blue')
#plt.scatter(x[idx2020],y1[idx2020],color='tab:blue',edgecolors='black',linewidths=2.5)


plt.ylim(bottom=0,top=13000)
plt.ylabel('Fires Reported')
plt.twinx()

plt.scatter(x,y2,c='tab:orange')
#plt.scatter(x[idx2020],y2[idx2020],c='tab:orange',edgecolors='black',linewidths=2.5)
#plt.plot(xp,p2(xp),linestyle=':',color='tab:orange')

plt.ylabel('Area Burned (Km$^2$)')
plt.ylim(bottom=0)
plt.xlim(left=1986,right=2023)

years = [1987,1994,2001,2008,2015,2022]
plt.xticks(years)


plt.arrow(1995,8400,-5,0,length_includes_head=True,width=300,head_length=1,edgecolor='None',facecolor='tab:blue')
plt.arrow(1990,7500,5,0,length_includes_head=True,width=300,head_length=1,edgecolor='None',facecolor='tab:orange')

plt.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.85, wspace=0.05, hspace=0.05)
#plt.show()
plt.savefig('Figure_1.png',dpi=600)
#plt.savefig('California_Wild_Fires.svg')