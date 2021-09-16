# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 08:51:14 2021

@author: lucaz
"""
import numpy as np
import bson
import json
import pandas as pd
import pylab as plt
import datetime
import scipy as sp
from scipy.signal import argrelextrema

path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/MindsphereFleetManager"
#%%
#read Bson file
with open(path+"/AQS_cycle10.bson", "rb") as rf:
    data = bson.decode(rf.read())
    
#se il file è phase ha variabili normali e phase, cioè due etichette nel dict, altrimenti solo 1    
datafr1=pd.DataFrame(data[list(data)[0]])
if len(list(data))==2:
    datafr_phase=pd.DataFrame(data[list(data)[1]])
#%%
#read excel file
datafr=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogFast.xlsx")
datafrsl=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogSlow.xlsx")
#%%
# read all the bson filename in the path
#funziona
import glob, os
files=[]
os.chdir(path)
for file in glob.glob("*.bson"):
    files.append(file)
    
#%%
#take time from iso 8601 data

def timefromiso(dataiso):
    return datetime.datetime.fromisoformat(dataiso[:-1])

#trasforma orario formato hh,mm,ss,micros in secondi 
def time_to_num(t):
    return(t.microsecond*10**(-6)+t.second+t.minute*60+t.hour*60*60+(t.day-1)*60*60*24)

#%%
def numbers_from_time(df,timename='_time'):
    timenum=[]
    #time0=df[timename][:1]
    
    #t0=time_to_num(timefromiso(time0.iloc[time0['Index']==0]))
    for time in df[timename]:
        t=timefromiso(time)
        timenum.append(time_to_num(t))
    return (np.array(timenum))#-timenum[0])

    #%%
with open('mapping_phasevariables.json') as json_file:
    data_phase=json.load(json_file)
phasevars=pd.DataFrame(data_phase['phaseIdVars'])



#%%

#mappo le due variabili, se una è il tempo solo variabile tempo, altrimenti faccio x-y, x-t e y-t
def mapping_var_big_than(df,namevar,namevar2,limit=None,timename='_time',title='correlation time-variables'):
   if limit==None:
       df1=df 
   else:
       df1=df.loc[df[namevar]>limit]
   y=df1[namevar]
   if (namevar2=='_time'):
       x= numbers_from_time(df1,namevar2)
       plt.figure()
       plt.scatter(x, df1[namevar],marker='.')
   else:
       time=numbers_from_time(df1,timename)
       x=df1[namevar2]
       fig, axs = plt.subplots(2, 2)
       fig.suptitle(title)
       axs[0,0].scatter(x, y,marker='.')
       axs[0,0].set_title(namevar+' - '+namevar2)
#       plt.xlabel(namevar2)
#       plt.ylabel(namevar)
#       plt.figure()
       axs[0,1].scatter(time, y,marker='.')
       axs[0,1].set_title(namevar+' - time')
#       plt.xlabel('time')
#       plt.ylabel(namevar)
#       plt.figure()
       axs[1,0].scatter(time, x,marker='.')
       axs[1,0].set_title(namevar2+' - time')
#       plt.xlabel('time')
#       plt.ylabel(namevar2)
##       
#%%

minima=[]   
minima=argrelextrema(np.array(datafr3['analogFast3']), np.less)
#%%
#add to df a column of time in second
def add_time_as_number(df,timename='_time'):
    timenumberdf=pd.DataFrame({"Time number":numbers_from_time(df,timename)})
    return(pd.concat([df,timenumberdf],axis=1))

#%%
#division of df analogFast about 1-9-2021 in cycles
df_1_9=add_time_as_number(datafr[:5343])
#timenumberdf=pd.DataFrame({"Time number":numbers_from_time(df_1_9,'_time')})
#df_1_9=pd.concat([df_1_9,timenumberdf],axis=1)

#timefs_1_9=numbers_from_time(df_1_9,'_time')
df_1_9_cut1=df_1_9[116:880]
df_1_9_cut2=df_1_9[1220:1574]
   
df_1_9_cut3=df_1_9[1585:2601]  
   
df_1_9_cut4=df_1_9[2680:5155]
df_1_9_cut5=df_1_9[5167:5247]
df_sl_1_9=add_time_as_number(datafrsl[:858])
#%%
   
   
df_1_9=datafr[:5343]
timefs_1_9=numbers_from_time(df_1_9,'_time')   
timesl_1_9=numbers_from_time(df_sl_1_9,'_time')
plt.figure()
plt.scatter(timesl_1_9,df_sl_1_9['analogSlow8'],marker='.')

plt.scatter(timefs_1_9,df_1_9['analogFast1'],marker='.')
plt.figure()
plt.scatter(timesl_1_9,df_sl_1_9['analogSlow8'],marker='.')

plt.scatter(timefs_1_9,df_1_9['analogFast3'],marker='.')

#%%
plt.figure()
plt.scatter(df_sl_1_9['Time number'],df_sl_1_9['analogSlow8'],marker='.')
plt.scatter(df_1_9['Time number'],df_1_9['analogFast1'],marker='.')
plt.scatter(df_1_9['Time number'],df_1_9['analogFast3'],marker='.')


#%%

CALCOLARE IL VOLUME DE


