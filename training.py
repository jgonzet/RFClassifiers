rpy  = ['roll_deg', 'pitch_deg', 'yaw_deg']
rpy_ref = ['rollref_deg','pitchref_deg','yawref_deg']
rpy_speed = ['rollspeed_degperrsec','pitchspeed_degperrsec','yawspeed_degperrsec']
fmot = ["f{}_kg".format(i+1) for i in range(6)]
zlmn = ['Z','L','M','N']

class Format:
    end = '\033[0m'
    underline = '\033[4m'
import pandas as pd
import numpy as np
import pathlib
import joblib
import sklearn.pipeline
import sklearn.ensemble
import time
#=============================================================================

#LEAFS = [7500,7000,6500,6000,5500,5000]
#for v,min_leaf_size in enumerate(LEAFS):

version = "1.2"

sims_train = 1000        # cantidad de simulaciones por cada rotor
n_e = 20
n = 9
fs = 100

# Parametros de cada arbol:
min_leaf_size = 9000
depth = 7
features_max = 4
ratio = 0.01
BSTP = False
pesos_features = [{0: ratio, 1: 1}, {0: ratio, 1: 1}, {0: ratio, 1: 1}, {0: ratio, 1: 1},{0: ratio, 1: 1},{0: ratio, 1: 1}]



state = 42 # / None

FEATURES = [rpy+rpy_ref+fmot]
names = ['rpy+rpy_ref+fmot']


Fs = int(np.floor(fs))
time_fail = 0.5
time_all = 10
time_cut = 0.2
rotores = 6
labelnumber = 6

datapoints = int(fs*time_all+1)
failpoints = int(fs*time_fail)
cutpoints = int(fs*time_cut)
factor = round(200/fs)



for k,features in enumerate(FEATURES):
    
    time_init = time.time()    
    print('------------------------------------------------------------------\n')
    print(Format.underline+'V{}:  {}'.format(version,time.ctime(time_init))+Format.end)
    print('Parametros: n={} a {}Hz, {} arboles, {} sims por motor.'.format(n,Fs,n_e,sims_train))
    print('Depth:{}, Min.Leaf_size:{}, Max.Features:{}.'.format(depth,min_leaf_size,features_max))
    print('Bootstrap:{},Class Weigth Ratio:{}.'.format(BSTP,ratio))
    #print(clf[1])
    
    len_features = len(features)    
    training_data = np.empty(((datapoints-cutpoints-n+1)*sims_train*rotores,n*len_features+labelnumber))
    
    dfs = []
    
    for r in range(rotores):
        
        ddir = pathlib.Path("../Data/Train/fail{}".format(r+1))   # me paro en la carpeta
        
        for s in range(sims_train):
            
            fn = ddir/"run{}.csv".format(s+1)                  # y busco el csv
            assert(fn.exists())
            dfs = pd.read_csv(fn)
            
            matrix = dfs[features].values[::factor,:]   #extraigo features, paso de dataframe a matriz float, y subsampleo    
            matrix = np.append(matrix, np.zeros((n-1,len_features)), axis=0)      #n-1 filas al final
            matrix = np.append(matrix, np.zeros((matrix.shape[0],(n-1)*len_features+labelnumber)), axis=1) #n-1 columnas de features + flags
    
            for j in range(n-1):#armo la ventana
                matrix[j+1:j+1+datapoints,len_features*(j+1):len_features*(j+2)] = matrix[:datapoints,:len_features]
    
            matrix = matrix[n-1:datapoints,:] # quito las primeras (n-1) filas y corto en datapoints            
            matrix[-failpoints:,-labelnumber+r] = 1 # seteo el flag de la falla en las ultimas muestras    
            matrix = np.append(matrix[:-failpoints,:],matrix[(-failpoints+cutpoints):,:],axis=0) #quito los cutpoints.
            
            training_data[(s+r*sims_train)*matrix.shape[0]:(s+1+r*sims_train)*matrix.shape[0],:] = matrix # meto el vuelo en el set de datos
    
    
    
    # In[3]: entrenamiento de un clasificador Forrest Tree.

    
    clf = sklearn.ensemble.RandomForestClassifier(bootstrap=BSTP,random_state=state,n_estimators = n_e,
                                                  min_samples_leaf=min_leaf_size,max_depth = depth,
                                                  max_features = features_max,class_weight=pesos_features)
    
    
    
    clf = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),clf) # Normaliza los features  
    #print(clf[1])

    clf.fit(training_data[:,:-labelnumber],training_data[:,-labelnumber:]) # Entrenamiento
    
    
    
    # In[4]: guardo el clasificador
    joblib.dump(clf, 'Clasificadores/RF({})_n{}_f{}_tr{}_{}'.format(names[k],n,Fs,sims_train,version))
    
    
    # In[5]: calculo el tiempo del entrenador:
    intervalo = time.time()-time_init   #TOC    
    horas = int(np.floor(intervalo/3600))
    minutos = int(np.floor((intervalo-horas*3600)/60))    
    segundos = round(intervalo - horas*3600 - minutos*60)
    print('Tiempo de entrenamiento para n{}_fs{}: {}:{}:{} (hh:mm:ss)'.format(n,Fs,horas,minutos,segundos))
    print('------------------------------------------------------------------')
   # print('Fin del entrenamiento: {}'.format(time.ctime(time.time())))
    print('\a')
