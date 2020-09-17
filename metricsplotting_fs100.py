# In[0]:

import numpy as np
import matplotlib.pyplot as plt


# In[1]:

n_e = 20

n = np.array([3,6,9,12,15])
fs=[100]
Fs = np.floor(fs)


# Para n variable:
threshold = np.arange(0,1,1/n_e)
Ps = np.zeros((n.shape[0],n_e+2))
Rs = np.zeros((n.shape[0],n_e+2))
falses = np.zeros((n.shape[0],n_e))
delays = np.zeros((n.shape[0],n_e))
dists_threshold = np.zeros((n.shape[0],n_e))
sigma_delays = np.zeros((n.shape[0],n_e))
areas = np.zeros((n.shape[0]))


for i in range(n.shape[0]):
    Ps[i,:]              = np.load('Results/Metricas/P_n{}_fs{}.npy'.format(n[i],int(Fs[0])))
    Rs[i,:]              = np.load('Results/Metricas/R_n{}_fs{}.npy'.format(n[i],int(Fs[0])))
    falses[i,:]          = np.load('Results/Metricas/falses_n{}_fs{}.npy'.format(n[i],int(Fs[0])))
    areas[i]             = np.load('Results/Metricas/area_n{}_fs{}.npy'.format(n[i],int(Fs[0])))
    delays[i,:]          = np.load('Results/Metricas/delay_n{}_fs{}.npy'.format(n[i],int(Fs[0])))
    sigma_delays[i,:]    = np.load('Results/Metricas/sigma_delay_n{}_fs{}.npy'.format(n[i],int(Fs[0])))


#PR curve
fig, ax = plt.subplots(figsize=(20,10))    
plt.plot(Rs.T, Ps.T) #,linestyle='--',marker='o'
plt.title('Precision-Recall', fontsize=25, color=np.array([0.2, 0.2, 0.7]))
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
axes = plt.gca()
plt.grid()
axes.set_ylim(0,1.1);
axes.set_xlim(0,1.1);
plt.legend(['n=3','n=6','n=9','n=12','n=15'])

# Enhanced
a = plt.axes([0.32, 0.3, .4, .4], facecolor='lightgrey')
plt.plot(Rs.T, Ps.T)
axes = plt.gca()
plt.grid()
axes.set_ylim(0.91,1.025);
axes.set_xlim(0.975,1.01);


plt.savefig('Results/Figures/PRcurves_fs100')


# Area curve
fig, ax = plt.subplots(figsize=(10, 10))    
plt.plot(n,areas)
plt.title('Precision-Recall areas', fontsize=25, color=np.array([0.2, 0.2, 0.7]))
plt.xlabel('n', fontsize=20)
plt.ylabel('Area [%]', fontsize=20)
plt.xticks(n)
axes = plt.gca()
plt.grid()
axes.set_xlim(np.min(n),np.max(n));
axes.set_ylim([99.4,100]);
plt.savefig('Results/Figures/PRarea_fs100')


# False positives vs precision curve
fig, ax = plt.subplots(figsize=(10,10))    
plt.plot( np.flip(falses,1).T,Ps[:,1:-1].T)
plt.title('False detections', fontsize=25, color=np.array([0.2, 0.2, 0.7]))
plt.ylabel('Precision', fontsize=20)
plt.xlabel('False detections per hour', fontsize=20)
axes = plt.gca()
plt.grid()
axes.set_xlim([0,10]);
axes.set_ylim(0,1);
plt.legend(['n=3','n=6','n=9','n=12','n=15'])
plt.savefig('Results/Figures/Falsedetections_fs100')



 #Delay curve
fig, ax = plt.subplots(figsize=(20, 15))
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
          
for i in range(n.shape[0]):    
    ax.plot(threshold,delays[i,:]+sigma_delays[i,:],color=colors[i],linestyle='dashed')
    ax.plot(threshold,delays[i,:]-sigma_delays[i,:],color=colors[i],linestyle='dashed')
    ax.plot(threshold,delays[i,:],color=colors[i],label="n = {}".format(n[i]))

plt.title('Delay in detection', fontsize=25, color=np.array([0.2, 0.2, 0.7]))
plt.xlabel(r'$\tau$', fontsize=50)
plt.ylabel('Delay [ms]', fontsize=50)
plt.xticks(threshold)
axes = plt.gca()
plt.grid()
axes.set_ylim([101,349]);
axes.set_xlim(0.26,0.99);
axes.tick_params(labelsize=30)
plt.legend(loc='best',fontsize='40')

#plt.savefig('Results/Figures/delays_fs100.eps',bbox_inches='tight',format='eps')
plt.savefig('Results/Figures/delays_fs100',bbox_inches='tight')