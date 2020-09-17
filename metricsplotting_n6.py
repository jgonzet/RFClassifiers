# In[0]:

import numpy as np
import matplotlib.pyplot as plt


# In[1]:

n_e = 20

n = np.array([6])
f = np.array([50,200/3,100,200])
Fs = np.floor(f)


threshold = np.arange(0,1,1/n_e)
Ps = np.zeros((f.shape[0],n_e+2))
Rs = np.zeros((f.shape[0],n_e+2))
falses = np.zeros((f.shape[0],n_e))
delays = np.zeros((f.shape[0],n_e))
maximos = np.zeros((f.shape[0],n_e))
minimos = np.zeros((f.shape[0],n_e))
dists_threshold = np.zeros((f.shape[0],n_e))
sigma_delays = np.zeros((f.shape[0],n_e))
areas = np.zeros((f.shape[0]))


for i,fs in enumerate(Fs): #voy a poner un clasificador por fila, con tantas columnas como tresholds se utilizo en metrics.py (n_e por lo general)
    Ps[i,:]              = np.load('Results/Metricas/P_n{}_fs{}.npy'.format(n[0],int(fs)))
    Rs[i,:]              = np.load('Results/Metricas/R_n{}_fs{}.npy'.format(n[0],int(fs)))
    falses[i,:]          = np.load('Results/Metricas/falses_n{}_fs{}.npy'.format(n[0],int(fs)))
    areas[i]             = np.load('Results/Metricas/area_n{}_fs{}.npy'.format(n[0],int(fs)))
    delays[i,:]          = np.load('Results/Metricas/delay_n{}_fs{}.npy'.format(n[0],int(fs)))
    sigma_delays[i,:]    = np.load('Results/Metricas/sigma_delay_n{}_fs{}.npy'.format(n[0],int(fs)))



# PR curve
fig, ax = plt.subplots(figsize=(20,10))    
plt.plot(Rs.T, Ps.T) #,linestyle='--',marker='o'
plt.title('Precision-Recall', fontsize=25, color=np.array([0.2, 0.2, 0.7]))
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
axes = plt.gca()
plt.grid()
axes.set_ylim(0,1.1);
axes.set_xlim(0,1.1);
plt.legend(['fs=50Hz','fs=66Hz','fs=100Hz','fs=200Hz'])

# Enhanced
a = plt.axes([0.32, 0.3, .4, .4], facecolor='lightgrey')
plt.plot(Rs.T, Ps.T)
axes = plt.gca()
plt.grid()
axes.set_ylim(0.91,1.025);
axes.set_xlim(0.975,1.01);

plt.legend(['fs=50Hz','fs=66Hz','fs=100Hz','fs=200Hz'])

plt.savefig('Results/Figures/PRcurves_n6')


# Area curve
fig, ax = plt.subplots(figsize=(10, 10))    
plt.plot(Fs,areas)
plt.title('Precision-Recall areas', fontsize=25, color=np.array([0.2, 0.2, 0.7]))
plt.xlabel('Fs', fontsize=20)
plt.ylabel('Area [%]', fontsize=20)
plt.xticks(f)
axes = plt.gca()
plt.grid()
axes.set_xlim(np.min(Fs),np.max(Fs));
axes.set_ylim([99.4,100]);

plt.savefig('Results/Figures/PRarea_n6')


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
plt.legend(['fs=50Hz','fs=66Hz','fs=100Hz','fs=200Hz'])

plt.savefig('Results/Figures/Falsedetections_n6')



# Delay curve
fig, ax = plt.subplots(figsize=(20, 15))
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
         
for i in range(f.shape[0]):    
    plt.plot(threshold,delays[i]+sigma_delays[i],color=colors[i],linestyle='dashed')
    plt.plot(threshold,delays[i]-sigma_delays[i],color=colors[i],linestyle='dashed')
    plt.plot(threshold,delays[i],color=colors[i],label="{} Hz".format(int(Fs[i])))

#plt.title('Delay in detection', fontsize=25, color=np.array([0.2, 0.2, 0.7]))
plt.xlabel(r'$\tau$', fontsize=50)
plt.ylabel('Delay [ms]', fontsize=50)
plt.xticks(threshold)
axes = plt.gca()
plt.grid()
axes.set_ylim([101,349]);
axes.set_xlim(0.26,0.99);
axes.tick_params(labelsize=30)
plt.legend(loc='best',fontsize='40')

#plt.savefig('Results/Figures/delays_n6.eps',bbox_inches='tight',format='eps')
plt.savefig('Results/Figures/delays_n6',bbox_inches='tight')