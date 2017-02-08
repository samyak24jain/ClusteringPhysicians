import pandas as pd
import numpy as np
from kmodes import kmodes, kprototypes
import matplotlib.pyplot as plt

df = pd.read_csv('835_837_NPI.csv')

df = df.fillna(0)

dfC61 = df[df['a.dx'].str.contains("C61") & df['a.procedure_code'] != 0] #Malignant neoplasm of prostate
dfC67 = df[df['a.dx'].str.contains("C67") & df['a.procedure_code'] != 0] #Malignant neoplasm of bladder
dfE291 = df[df['a.dx'].str.contains("E29.1") & df['a.procedure_code'] != 0] #Testicular hypofunction
dfN200 = df[df['a.dx'].str.contains("N20.0") & df['a.procedure_code'] != 0] #Calculus of kidney
dfN201 = df[df['a.dx'].str.contains("N20.1") & df['a.procedure_code'] != 0] #Calculus of ureter

features = ['a.pat_birthday', 'a.procedure_code']
x = dfC61[features]

km = kmodes.KModes(n_clusters=4, init='Cao', n_init=10, verbose=1)

clusters = km.fit_predict(x)
centroids = km.cluster_centroids_

plt.scatter(x[features[0]], x[features[1]],s=60 ,c=clusters)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', color='y', s=169, linewidths=3)
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.show()
