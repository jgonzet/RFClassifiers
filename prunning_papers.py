rpy  = ['roll_deg', 'pitch_deg', 'yaw_deg']
rpy_ref = ['rollref_deg','pitchref_deg','yawref_deg']
rpy_speed = ['rollspeed_degperrsec','pitchspeed_degperrsec','yawspeed_degperrsec']
fmot = ["f{}_kg".format(i+1) for i in range(6)]
zlmn = ['Z','L','M','N']
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import seaborn as sns


fs = 100
n = 15
n_e = 20
Fs = int(np.floor(fs))




name = 'rpy+rpy_ref+fmot'
features = rpy+rpy_ref+fmot
len_features = len(features)
total_features = len_features*n

clf = joblib.load('Clasificadores/RF_n{}_f{}'.format(n,Fs))


clasificador = clf[1]
x=np.zeros((1,total_features))
for i in range(n_e):
    estimator = clasificador.estimators_[i]
    x = x+ estimator.feature_importances_
    
    # Graficamos un arbol:
    export_graphviz(estimator, out_file='tree.dot',rounded = True, proportion = False, precision = 2, filled = True)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree{}_n{}_f{}.png'.format(i+1,n,Fs), '-Gdpi=50'])
    from IPython.display import Image
    Image(filename = 'tree{}_n{}_f{}.png'.format(i+1,n,Fs))
    
argumentos = ((np.transpose( np.argsort(-x))))

for i in range(x.shape[1]):
    
    delay = np.floor(argumentos[i]/len_features)
    variable = int(argumentos[i] - delay*len_features),



#Mapa de calor:
 
salidas = pd.DataFrame(x.reshape(len_features,n,order='F'))#,index=timestamps)
fig,ax = plt.subplots(figsize=(10,7))
plt.title('Mapa de calor n={},{}Hz'.format(n,Fs), fontsize=30)
sns.heatmap(salidas.T, xticklabels=features,cmap="viridis",ax=ax,vmin=0, vmax=1)
ax.set_ylabel('Delay', fontsize=20)

plt.savefig('Mapas/Mapa de calor n{}_f{}'.format(n,Fs))#,format='eps')
