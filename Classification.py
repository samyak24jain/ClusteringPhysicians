# This code classifies the prostate cancer patients from the dataset 835_837_NPI.csv into 4 stages :
# Early Stage, Early Stage Intervention, Late Stage, Very Late Stage.
# It also fits the decision tree model to the data.

from __future__ import print_function
from math import floor
import os
import subprocess
import pandas as pd
import numpy as np
import pydot
import datetime
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image
from sklearn.externals.six import StringIO
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


#Reading 835_837_NPI.csv file
df=pd.read_csv("835_837_NPI.csv")

df = df.fillna(0)

#Filtering out only prostate cancer patients from the csv
df185=df[df['a.dx'].str.contains("185")]
dfC61=df[df['a.dx'].str.contains("C61")]
dfAppended = df185.append(dfC61)

target_values = {
    0 : "Early Stage", 
    1 : "Early Stage Intervention", 
    2 : "Late Stage",
    3 : "Very Late Stage"
}

targets = []
targets_encoded = []
dx = dfAppended['a.dx']
ndc_code = dfAppended['a.ndc_code']
proc_code = dfAppended['a.procedure_code']
strs = ['C79', 'C78', '197', '198']
es_pc = ['96402', '96413', 'J9155', 'J9171', 'J9217', 'J3489']
es_nc = ['00074228203', '00074368303', '00310095130', '52544015302', '55566840301', '62935045245', '62935045345','00024060545',
'00074334603','52544009276','52544015602','62935030330','62935075275','00074210803','00074366303','00024061030',
'00074347303','52544015402','55566830301','62935075375','00024022205','00024079375','00074364203',
'00310095036','62935022305','00069914100','00069914200','00069914400','00075800300','00075800400','00409020100','00703572000',
'00703573000','00955102000','00955102100','16714046500','16714050000','16729012000','16729022800','16729023100','16729026700',
'47335028500','63739093200','63739097100','45963073400','45963076500','45963078100','45963079000','66758005000','66758095000',
'42367012100','43598025800','43598025900','25021022200','00024582400','00024582300','00024582400']

esi_pc = ['77401','77402','77403','77404','77406','77407','77408','77409',
'77411','77412','77413','77414','77416','G6003','G6004','G6005','G6006','G6007',
'G6008','G6009','G6010','G6011','G6012','G6013','G6014','77418','G6015'
'55810','55812','55815','55821','55831','55840','55842','55845','55866','52650','52612','52614']


#Classifying the prostate cancer patients into various stages - Early, Late, etc. based on Prostate_Segmentation.xlsx

for i in range(len(dfAppended)):
    if any(x in dx.iloc[i] for x in strs):
        targets.append(target_values[3])
        targets_encoded.append(3)
    elif any(x in str(proc_code.iloc[i]) for x in es_pc) or any(x in str(ndc_code.iloc[i]) for x in es_nc):
        targets.append(target_values[2])
        targets_encoded.append(2)
    else:
        if any(x in str(proc_code.iloc[i]) for x in esi_pc):
            targets.append(target_values[1])
            targets_encoded.append(1)
        else:
            targets.append(target_values[0])
            targets_encoded.append(0)


target_df = pd.DataFrame({'Target' : targets, 'Target_Encoded' : targets_encoded})

dfAppended.reset_index(drop=True, inplace=True)
target_df.reset_index(drop=True, inplace=True)

#Adding 2 columns to the prostate cancer data - Target (The class like Early stage, etc.) and Target_Encoded (Encoded class)
result = dfAppended.join(target_df)

#Writing the new dataframe to a new csv file.
result.to_csv('ClassifiedData.csv')

now = datetime.datetime.now()
pat_age = []
pat_gender_code = []
pat_bday = result['a.pat_birthday']
_c22 = result['_c22']

for i in range(len(result)):
    pat_age.append(now.year - pat_bday.iloc[i])
    if _c22.iloc[i] == 'Male':
        pat_gender_code.append(0)
    else:
        pat_gender_code.append(1)

blahdf = pd.DataFrame({'pat_age' : pat_age, 'pat_gender_code' : pat_gender_code})
blahdf.reset_index(drop=True, inplace=True)
result.reset_index(drop=True, inplace=True)
result = result.join(blahdf)


#Splitting the data into training and testing sets
train_data = pd.DataFrame()
test_data = pd.DataFrame()

l = result.iloc[np.random.RandomState(40).permutation(len(result))]

#Doing a 80-20 split of the dataset for training and testing respectively
split80 = int(floor(0.8 * len(result))) 
split20 = int(len(result) - split80)

test_data = l[:split20] 
train_data = l[split20:]

#Features used in the model
features = ['pat_age', 'a.ndc_code', 'a.procedure_code']

x = train_data[features]
y = train_data['Target_Encoded']


#Fitting the Decision Tree model
#Can fine tune the parameters to improve accuracy of the model.
dt = DecisionTreeClassifier(min_samples_split=5, random_state=100, min_samples_leaf=200)
clf = dt.fit(x, y)

newtest = pd.DataFrame(test_data[features])
actual = pd.DataFrame(test_data['Target_Encoded'])

accuracy = clf.score(newtest, actual)
print ("Accuracy of Decision Tree Classifier = " + str(accuracy))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                         feature_names=features,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())

#Writing the Decision Tree Image as a jpeg
graph.write_jpeg("DecisionTreeClassifier.jpeg")