import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
fmot = ["f{}_kg".format(i+1) for i in range(6)]
rpy = ['roll_deg', 'pitch_deg', 'yaw_deg']

features = fmot
init = 0
slice=np.linspace(init,init+19,20)
#slice=[1]

for i in slice:
    ddir = pathlib.Path("Data/Train/fail4/run{}.csv".format(int(i+1)))
    #ddir = pathlib.Path("Data/vuelos reales/test{}.csv".format(int(i+1)))
    assert(ddir.exists())

    dfs = pd.read_csv(ddir).set_index("time_sec")
    #dfs = pd.read_csv(ddir)
    matrix = dfs[features].values
    timestamps = dfs.index

    plt.figure(figsize=(20,10))
   # plt.axis([0,30,0.2,0.5])
  #  plt.axis([0,30,-30,30])
    plt.plot(timestamps.values,matrix)
    plt.title("Muestra {}".format(int(i+1)))
    plt.gca().legend(('f1/roll','f2/pitch','f3/yaw','f4','f5','f6'))