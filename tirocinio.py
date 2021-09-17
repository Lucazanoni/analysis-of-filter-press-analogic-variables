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


path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/MindsphereFleetManager"
#%%
#read Bson file
with open(path+"/AQS_cycle10002_phase1.bson", "rb") as rf:
    data = bson.decode(rf.read())
    
#se il file è phase ha variabili normali e phase, cioè due etichette nel dict, altrimenti solo 1    
datafr1=pd.DataFrame(data[list(data)[0]])
if len(list(data))==2:
    datafr_phase=pd.DataFrame(data[list(data)[1]])
    
    #%%
with open('mapping_phasevariables.json') as json_file:
    data_phase=json.load(json_file)
phasevars=pd.DataFrame(data_phase['phaseIdVars'])

#%%
# read all the bson filename in the path
#funziona
import glob, os
files=[]
os.chdir(path)
for file in glob.glob("*.bson"):
    files.append(file)
    
#%%
#read excel file
df_fast=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogFast.xlsx")
df_slow=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogSlow.xlsx")

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
    if not(timename in df.columns):
        raise TypeError(timename+' do not exist in the dataframe')
    #time0=df[timename][:1]
    
    #t0=time_to_num(timefromiso(time0.iloc[time0['Index']==0]))
    for time in df[timename]:
        t=timefromiso(time)
        timenum.append(time_to_num(t))
    return (np.array(timenum))#-timenum[0])



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
#add to df a column of time in second
def add_time_as_number(df,timename='_time'):
    if not(timename in df.columns):
        raise TypeError(timename+' do not exist in the dataframe')
    timenumberdf=pd.DataFrame({"Time number":numbers_from_time(df,timename)})
    return(pd.concat([df,timenumberdf],axis=1))
#%%

#calc volume as itegral of flow

def volume_from_flow(flows,times):
    volume=[]
    volume.append(0)
    for i in range(0,len(flows)-1):
        volume.append(volume[i]+(flows[i+1]+flows[i])*(times[i+1]-times[i])/(2*3600))
    return volume

#%%
    """voglio selezionare una zona di punti, idealmente quelli di un ciclo, che sanno sopra un certo valore eludendo le fluttuazioni"""
def selecting_cycle(arr, limit, n_point_under,n_point_over):
    #arr=array of values
    #limit=lower limit in which points are not good
    #n_point_under=number of point under which no start cycle index is taken
    #n_point_over=number of point under which no end cycle index is taken
    #return=starting and ending cycle indices 
        if(n_point_under<1):
            raise TypeError("n_point_under should be at least 1")
        if(n_point_over<1):
            raise TypeError("n_point_over should be at least 1")
        if(n_point_under<len(arr)):
            raise TypeError("n_point_under could not be more than the length of arr")
        if(n_point_over<len(arr)):
            raise TypeError("n_point_over could not be more than the length of arr")
        if len(arr)<1:
            raise TypeError("arr is empty")
        indices=[]
        index_start=0
        index_end=0
        
        #        h=False
        k=False
        count1=0 #count1 count the numbers over limit
        count2=0
        for i in range(0,len(arr)):
            if arr[i]>limit:
        #                h=True
                count1=count1+1
                count2=0
            else:
        #                h=False
                count1=0
                count2=count2+1
            if (count1==n_point_under  and not(k)):
                index_start=i-n_point_under
                k=True
            if (count2==n_point_over and k):
                index_end=i-n_point_over
                indices.append([index_start,index_end])
                index_start=0
                index_end=0
                k=False
        return indices
    #da correggere

#%%
"""linear interpolation"""
"""np.interp(time1,time2, density)"""
#%%
#pairing slow and fast variable adding missing points in slow variables
#slow variables have a minor sampling rate, so we need to interpolate the missing points
#how? or linear interpolation or finding some low of the points and generating missing points from those curves
"""def linear_interpolation(x1,y1,x2,y2,xp):
    return ((xp-x2)/(x1-x2)*y1-(xp-x1)/(x1-x2)*y2)


def linear_variables_interpolation(var_array1,time1,var_array2,time2):#var_array1=more dense, var_array2= less dense
    var2_interpolated=np.zeros(len(time1))
    arr_same_values=np.in1d(time1,time2)#boolean array with True if the element of time1 is in time2
    for i in range(0,len(time1)):
        if arr_same_values[i]:
            var2_interpolated[i]=var_array2[i] #filled var2_interpolated with the values of time1 that are present in time2
#            
#    index_prev_true=0
#    index_next_true=0
#    #find the previous and next values in common between time 1 and 2 end interpolate points between the two
#    for i in range(1,len(time1)):
#        if arr_same_values[i]: 
#            index_prev_true=i
#        j=0
#        if (arr_same_values[i]):
#            j=i
#            while (not(arr_same_values[j])and (j<len(time1))):
#                j=j+1
#            if(j<len(time1)):
#                index_next_true=j
#                #here i will use also all the other interpolation curve
#                x1=time1[index_prev_true]
#                y1=var_array1[index_prev_true]
#                x2=time1[index_next_true]
#                y2=var_array1[index_next_true]
#                var2_interpolated[i]=linear_interpolation(x1,y1,x2,y2,time2[i])
#            else:
#                var2_interpolated[i]=0
    return var2_interpolated
        """
        
"""BASTAVA UNA FUNZIONE DI NUMPY"""
"""np.interp(time1,time2, density)"""
            
            


































#%%
"""
PROVE FUNZIONI E PER CAPIRE I DATI

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

fig, axs = plt.subplots(3, 1)
axs[0].scatter(datafr1['Time number'],datafr1['Analogs.analog1'],marker='.')
axs[1].scatter(datafr1['Analogs.analog1'],datafr1['Analogs.analog3'],marker='.')
axs[2].scatter(datafr1['Time number'],datafr1['Analogs.analog3'],marker='.')

"""

