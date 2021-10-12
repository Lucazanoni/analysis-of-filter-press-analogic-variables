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
import dateutil.parser
import warnings
import scipy as sp
import statistics as st
import glob, os
import statsmodels
from statsmodels.tsa.stattools import acf
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
path1="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/15 settembre- 4 ottobre dati/bson"
path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/MindsphereFleetManager"
path2="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/15 settembre- 4 ottobre dati/power"

#%%
#read excel file
#os.chdir(path)
#df_fast=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogFast.xlsx")
#df_slow=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_AnalogSlow.xlsx")
#df_power=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_Power.xlsx")
#df_power1=pd.read_excel(path+"/excel_csv/c6378abbfb4b4c079a63dd5489bed1e6_Power (1).xlsx")
#%%
analog_not_measured=['Analogs.analog2','Analogs.analog4','Analogs.analog5','Analogs.analog6',
                     'Analogs.analog7','Analogs.analog8','Analogs.analog9','Analogs.analog10',
                     'Analogs.analog11','Analogs.analog12','Analogs.analog17','Analogs.analog19',
                     'Analogs.analog23','Analogs.analog30','Analogs.analog31','Analogs.analog32',
                     'Analogs.analog33','Analogs.analog34','Analogs.analog35','Analogs.analog36',
                     'Analogs.analog37','Analogs.analog38','Analogs.analog39','Analogs.analog40']

analog_process=['Analogs.analog1','Analogs.analog3','Analogs.analog20','Analogs.analog29']
#%%
"""READING AND OPENING FILES JSON AND BSON"""

"""read all the filenames in the given path and return the list of filenames""" 
def read_json_names(path):
    files=[]
    os.chdir(path)
    for file in glob.glob("*.json"):
        files.append(file)
    return files
"""create a global dataframe (pandas) for each json file read"""
def json_file_to_df(filenames, path):
    os.chdir(path)
    for file in filenames:
        file1=file[33:-5]
        file1=file1.replace(" ","")
        file1=file1.replace("(","_")
        file1=file1.replace(")","")
        globals()[file1]=pd.read_json(file)

"""create a variable for phase of interst with the name of all the file of that phase """
"""number of phases=array with the numbers of the phases of interest"""
"""example:  make_bson_list_for_phase(path_prova,[1,2,3])"""
def make_bson_list_for_phase(path,numbers_of_phases):
    files=[]
    os.chdir(path)
    for file in glob.glob("*.bson"):
        files.append(file)
    for i in numbers_of_phases:
        names=[]
        for file in files:
            if file[12:18]==('phase'+str(i)):
                names.append(file)
            if file[13:19]==('phase'+str(i)):
                names.append(file)
        globals()['phase'+str(i)]=names
    """generate a dataframe pandas from bson file"""    
def df_from_bson(filename,path):
    with open(path+'/'+filename, "rb") as rf:
        data = bson.decode(rf.read())
    df=pd.DataFrame(data[list(data)[0]])
    df.dropna(subset = ['timestamp'], inplace=True)
    return df
"""generate a dataframe pandas from bson file for phase variable, in whing there are 2 db, one with analogs and another with phase"""
def df_from_phase_bson(filename, path):
    with open(path+'/'+filename, "rb") as rf:
        data = bson.decode(rf.read())
    df=pd.DataFrame(data[list(data)[0]])
    df.dropna(subset = ['timestamp'], inplace=True)
    df_phase=pd.DataFrame(data[list(data)[1]])
    df_phase.dropna(subset = ['timestamp'], inplace=True)
    return df,df_phase                
   
    
"""select all file in path that has a certain number in position 9-10, like AQS_cycle62_phase2.... has number 62"""
def cycle_list_file(n_cycle,path):
    files=[]
    os.chdir(path)
    for file in glob.glob("*.bson"):
        files.append(file)
    names=[]
    for file in files:
        if file[9:11]==str(n_cycle):
            names.append(file)
            
    return names
"""take the max of a certain phase of each cycle foreach of the variable of interest"""
def max_of_phase(files, path,variables):
    x=np.zeros([len(files),len(variables)])
    time=[]
    j=0
    for file in files:
        x1=np.zeros(len(variables))
        df,df_phase=df_from_phase_bson(file, path)
        i=0
        for var in variables:
            x1[i]=max(np.array(df_phase[var]))
            i=i+1
        x[j,:]=x1
        df_phase=add_time_as_timeseries(df_phase)
        time.append(df_phase['timeserie'][len(df_phase)-1])
        j=j+1
    sorted_index = np.argsort(time)
    x1=np.zeros([len(files),len(variables)])
    time = [time[i] for i in sorted_index]
    j=0
    for i in sorted_index:
        x1[j,:]=x[i,:]
        j=j+1
    return x1,time
#%%
"""calculate the parameters of a inear regression of time/volume over volume
   in theory this relation should be linear if the pressure is constant, and after few seconds in phase 3 othe cycles it is true"""
   
def time_volume_over_volume(filenames, path,figure=False):
    pressure='Analogs.analog3'
    slopes=[]
    intercepts=[]
    err=[]
    r_values=[]
    time=[]
    mean_pressure=[]
    st_err_pressure=[]
    i=0
    for file in filenames:
        i=i+1
        df,df_phase=df_from_phase_bson(file, path)
        df=add_time_as_number(df,'timestamp')
        limit=max(np.array(df_phase['PhaseVars.phaseVariable4']))-1 #limit over which the pressure remain constant
        res = next(x for x, val in enumerate(df[pressure]) if val > limit)
        res2 = next(x for x, val in enumerate(df[pressure][res:]) if val <limit-2 ) #when pressure drops in the last seconds it cannot be more considered constant
        mean_pressure.append(np.mean(df[pressure][res:res2]))
        st_err_pressure.append(np.std(df[pressure][res:res2]))
        #the volume is the cumulative volume, so i cannot ignore the volume already present in the filter
        volume=volume_from_flow(np.array(df['Analogs.analog1']),np.array(df['Time number']))#+df_phase['PhaseVars.phaseVariable6'][0]
        #calculate T over V in the region of costant pressure 
        t_V=(df['Time number'][res:res2]-df['Time number'][res])/volume[res:res2]
        
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(volume[res:res2],t_V)
        if figure:
            plt.figure()
            plt.scatter(volume[res:res2],t_V,marker='.',c=df['Time number'][res:res2])
            plt.colorbar()
            plt.title('t/V - V --- colormaps of time(s)' )
            print('r_value = ',r_value)
            x=np.linspace(min(volume[res:res2]),max(volume[res:res2]),100)
            plt.plot(x,intercept+slope*x,linestyle='--')
#            plt.savefig(fname="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/risultati/t_V over V/figure/figure_"+str(i)+".png")
#            plt.close()

            
        slopes.append(slope)
        intercepts.append(intercept)
        err.append(std_err)
        r_values.append(r_value)
        time.append(timefromiso(df['timestamp'][res]))
    return slopes,intercepts,err,r_values,time


#%%
def volume_density_bson(files,path):
    volume=[]
    density=[]
    for file in files:
        df,df_phase=df_from_phase_bson(file, path)
        volume.append(max(np.array(df_phase['PhaseVars.phaseVariable6'])))
        density.append(max(np.array(df_phase['PhaseVars.phaseVariable9'])))
    plt.scatter(volume,density)
    return(volume,density)
def time_density_feeding_bson(files,path):
    time=[]
    density=[]
    finalfeeding=[]
    initialfeeding=[]
    for file in files:
        df,df_phase=df_from_phase_bson(file, path)
        time.append(max(np.array(df_phase['PhaseVars.phaseVariable1'])))
        density.append(max(np.array(df_phase['PhaseVars.phaseVariable9'])))
        finalfeeding.append(max(np.array(df_phase['PhaseVars.phaseVariable3'])))
        initialfeeding.append(max(np.array(df_phase['PhaseVars.phaseVariable2'])))
    return(time,initialfeeding,finalfeeding,density)

def selected_time_density(time,density,initialfeeding,finalfeeding,limits_initialfeedig=[200,230],limits_final=[3,10]):
    d=[]
    t=[]
    in_fed=[]
    fin_fed=[]
    for i in range(len(time)):
        if(initialfeeding[i]>=limits_initialfeedig[0] and initialfeeding[i]<=limits_initialfeedig[1] 
        and finalfeeding[i]>=limits_final[0] and finalfeeding[i]<=limits_final[1]):
            t.append(time[i])
            d.append(density[i])
            in_fed.append(initialfeeding[i])
            fin_fed.append(finalfeeding[i])
    x=np.zeros([len(d),4])
    x[:,0]=t
    x[:,1]=in_fed
    x[:,2]=fin_fed
    x[:,3]=d
    return x
"""calculate the daily mean of a cartain variable"""
def mean_by_day(arr,dates):
    day=dates[0].day
    means=[]
    meanday=[]
    st_err=[]
    dates2=[]
    for i in range(len(dates)):
        if dates[i].day==day:
            meanday.append(arr[i])
        if dates[i].day!=day:
            means.append(np.mean(meanday))
            st_err.append(np.std(meanday))
            meanday=[]
            meanday.append(arr[i])
            dates2.append(datetime.datetime(dates[i-1].year,dates[i-1].month,dates[i-1].day))
            day=dates[i].day
        if i==(len(dates)-1):
            means.append(np.mean(meanday))
            st_err.append(np.std(meanday))
            dates2.append(datetime.datetime(dates[i].year,dates[i].month,dates[i].day))
    return means,st_err,dates2
            
#%%
def func(x,a,b):
    return x*a+b

from scipy.optimize import curve_fit
#from pylab import *
def linear_fit(x,y,banderror=False):
    x=np.array(x)
    y=np.array(y)
    popt, pcov = curve_fit(func, x,y)
    perr=np.sqrt(np.diag(pcov))
    popt_up=popt+perr
    popt_dw=popt-perr
    fit = func(x, *popt)
    fit_up = func(x, *popt_up)
    fit_dw = func(x, *popt_dw)
    fig, ax = plt.subplots(1)
    plt.scatter(x, y, label='data')
    
    sorted_index = np.argsort(x)
    fit_up = [fit_up[i] for i in sorted_index]
    fit_dw = [fit_dw[i] for i in sorted_index]
    plt.plot(x, fit, 'r', lw=2, label='best fit curve')
    if banderror:
        ax.fill_between(np.sort(x), fit_up, fit_dw, alpha=.25, label='1-sigma interval')
    
    return popt,perr


def multiple_linear_regression(x,y):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    LR = LinearRegression() 
    LR.fit(x_train,y_train)
    y_prediction =  LR.predict(x_test)
    score=r2_score(y_test,y_prediction)
    print('r2 score is ',score)
    print('mean_sqrd_error is==',mean_squared_error(y_test,y_prediction))
    print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_prediction)))
    return LR

#%%
def poly_fit(x,y,deg1=10):
    z=np.polyfit(x, y,deg1)
    p=np.poly1d(z)
    xp = np.linspace(min(x),max(x) , len(y))
    _ = plt.plot(x, y, '.', xp, p(xp), '-')

    
"""fitting values of a curve as a*e^(-b*x^c)+d""" 
def neg_exp(x,a,b,c,d):
    return a*np.exp(-b*(x**c))+d
def exp_fit(x,y):
    x=np.array(x)
    y=np.array(y)
    popt, pcov = curve_fit(neg_exp, x,y,p0=[160,0.16,0.5,10])#p0 take from a previous fit with no p0 parameter
    perr=np.sqrt(np.diag(pcov))
    fit = neg_exp(x, *popt)
    plt.scatter(x, y, label='data',marker='.')   
    plt.plot(x, fit, 'r', lw=2, label='best fit curve')
    return popt,perr


def fitting_values_feeding_law(filenames,path,figure=False):
    fit_values=[]
    fit_err=[]
    for file in filenames:
        if figure:
            plt.figure()
        df,df_phase=df_from_phase_bson(file,path)
        df=add_time_as_number(df,'timestamp')
        x=df['Time number']
        y=df['Analogs.analog1']
        val,err=exp_fit(x,y)
        fit_values.append(val)
        fit_err.append(err)
    return fit_values,fit_err

#%%
"""analysis of feeding and pressure in a single phase""" 

def feeding_volume_in_a_phase(phase_file,path):
    df,df_phase=df_from_phase_bson(phase_file,path)
    df=add_time_as_number(df[['timestamp','Analogs.analog1']],'timestamp')
    volume=volume_from_flow(df['Analogs.analog1'],df['Time number'])
    plt.figure()
    plt.scatter(volume[20:-5],1/df['Analogs.analog1'][20:-5])
    
    
#%%    
"""little pipeline"""
"""
df2,df_p_var2=df_from_phase_bson(phase2[12],path1)
df3,df_p_var3=df_from_phase_bson(phase3[12],path1)
for index1 in df2.columns:
    index2 =df2.columns[1]
    if (index1 not in analog_not_measured and index2 not in analog_not_measured and index1!='timestamp' and index2!='timestamp'):
        plt.figure()
        plt.scatter(df2[index1],df2[index2],marker='.')
        plt.title(index1+'-'+index2)"""
#%%
    
"""TIME IN FORM OF NUMBER FROM DATES ISO OR DATETIME FORMAT"""
#take data from iso data
def take_datetime(df,timename='_time'):
    dates=[]
    for date in df[timename]:
        date=dateutil.parser.isoparse(date[:-1])
        dates.append(date)
    date=pd.DataFrame({'time':dates})
    return date

#take time from iso 8601 data
def timefromiso(dataiso):
    return datetime.datetime.fromisoformat(dataiso[:-1])

#trasforma orario formato hh,mm,ss,micros in secondi 
def time_to_num(t):
    return(t.microsecond*10**(-6)+t.second+t.minute*60+t.hour*60*60+(t.day-1)*60*60*24+(t.month-1)*60*60*24*30)
    
#add to df a column of time in second
"""if time column is in iso format"""  
def add_time_as_number(df,timename='_time'):
    if not(timename in df.columns):
        raise TypeError(timename+' do not exist in the dataframe')
    timenumberdf=pd.DataFrame({"Time number":numbers_from_time(df,timename)})
    return(pd.concat([df,timenumberdf],axis=1))
    
"""if time column is in datetime format"""    
def add_time_as_number2(df,timename='_time'):
    timenum=[]
    t0=df[timename].iloc[0]
    t0=time_to_num(t0)
    for time in df[timename]:
        timenum.append(time_to_num(time)-t0)
    timenumberdf=pd.DataFrame({"Time number":timenum})
    return pd.concat([df,timenumberdf],axis=1)

def numbers_from_time(df,timename='_time'):
    timenum=[]
    if not(timename in df.columns):
        raise TypeError(timename+' do not exist in the dataframe')
    
    t0=time_to_num(timefromiso(df[timename][0]))
    for time in df[timename]:
        t=timefromiso(time)
        timenum.append(time_to_num(t)-t0)
    return (np.array(timenum))

def add_time_as_timeseries(df,timename='timestamp'):
    timeserie=[]
    for time in df[timename]:
        timeserie.append(datetime.datetime.fromisoformat(time[:-1]))
    timeserie=pd.DataFrame({'timeserie':timeserie})
    return pd.concat([df,timeserie],axis=1)
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

#calc volume as itegral of flow

def volume_from_flow(flows,times):
    """flow must be an array"""
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
        if(n_point_under>len(arr)):
            raise TypeError("n_point_under could not be more than the length of arr")
        if(n_point_over>len(arr)):
            raise TypeError("n_point_over could not be more than the length of arr")
        if len(arr)<1:
            raise TypeError("arr is empty")
        indices=[]
        index_start=0
        index_end=0
        

        k=False #k is a variable of status
        count1=0 #count1 count the numbers over limit
        count2=0 #count1 count the numbers under limit
        for i in range(0,len(arr)):
            if arr[i]>limit:

                count1=count1+1
                count2=0
            else:

                count1=0
                count2=count2+1
            if (count1==n_point_under  and not(k)):
                index_start=i-n_point_under+1
                k=True
            if (count2==n_point_over and k):
                index_end=i-n_point_over
                indices.append([index_start,index_end])
                index_start=0
                index_end=0
                k=False
        return indices

"""con  n_point_under=3, e  n_point_over=5 funziona abbastanza bene"""

#%%

def plot_2_var_interpoled(var1,var2,time1,time2):
    var2_interp=np.interp(time1,time2,var2)
    plt.figure()
    plt.scatter(time1,var2_interp,marker='.')
    plt.scatter(time2,var2,marker='o',alpha=0.2)
    plt.figure()
    plt.scatter(var2_interp,var1)

#%%
"""scattr plot of all division of the df"""
def scatter_plot_cycles(df,indices,y_var,x_var='Time number'):
    for index in indices:
        plt.figure()
        plt.scatter(df[x_var][index[0]:index[1]],df[y_var][index[0]:index[1]],marker='.')
        
        
#%%
"""analisi densità-volume"""
"""density,time and flow should be array"""
def density_volume(density,flows,time1,time2):
    index_flow=selecting_cycle(flows,1.,3,5) #divide the points of flow in cycles in which flow is not null avoiding fluctuations
    density_interp=np.interp(time1,time2,density) #linear interpolation of density (less sample) with the time of flow (more sample)
    """time1 is the time whit higher number of point, time 2 less point"""
    volumes=[]
    densities=[]
    for index in index_flow:
        volume=volume_from_flow(flows[index[0]:index[1]],time1[index[0]:index[1]]) #calulate volumes as integral of flow in time
        volumes.append(volume)
        densities.append(density_interp[index[0]:index[1]])
        plt.figure()
        plt.scatter(volume,density_interp[index[0]:index[1]] , marker='.')
#        plt.figure()
#        plt.scatter(time1[index[0]:index[1]],volume, marker='.')
    return(densities, volumes)
#%%
def integrals_of_cycles(arr,time,cycle_index):
    total_integral=np.zeros(len(cycle_index))
    time_integral=np.zeros(len(arr))
    j=0
    for index in cycle_index:
        total_integral[j]=np.trapz(arr[index[0]:index[1]],x=time[index[0]:index[1]])
        j=j+1
        for i in range(index[0],index[1]):
            if i==0:
                time_integral[i]=0    
            time_integral[i]=time_integral[i-1]+np.trapz([arr[i-1],arr[i]],x=[time[i-1],time[i]])
    return total_integral,time_integral

#%%
"""divide in cycle using total feeding time=analog 21"""
#
def select_cycle2_time(df,var,timename='Time number'):
    j=0 
    #j= variabile di ciclo: 0= ciclo spento, 1=ciclo acceso
    start_i=[]
    end_i=[]
    times_start=[]
    times_end=[]
    for i in range(0,len(df[var])):
        if (df[var][i]!=0 and not(np.isnan(df[var][i])) and j==0):
            start_i.append(i)
            j=1
        if ((df[var][i]==0 or np.isnan(df[var][i])) and j==1):
            end_i.append(i)
            j=0
    for i in range (0,len(end_i)):
        times_start.append(df[timename][start_i[i]])
        times_end.append(df[timename][end_i[i]])
    return (times_start,times_end)



def select_cycles_indices_by_time(df, times_start,times_end, timename='Time number'):
    indices=[]
    for i in range(0,len(times_start)):
        df1=df.loc[df[timename]<=times_end[i]]
        df1=df1.loc[df1[timename]>=times_start[i]]
        indices.append([df1.index[0]+1,df1.index[len(df1)-1]+1])
    return indices

#%%

def final_feeding_delivery_on_pressure(df,indices,limit_pressure):
    feed_var='analogFast1'
    pressure_var='analogFast3'
    for i in range(0,indices[1]-indices[0]):
        j=indices[1]-i
        if df[pressure_var][j]>limit_pressure:
            return df[feed_var][j-5]

#%%
            
def linear_regression_assumptions(features, label, feature_names=None):
    """
    Tests a linear regression on the model to see if assumptions are being met
    """
    from sklearn.linear_model import LinearRegression
    
    # Setting feature names to x1, x2, x3, etc. if they are not defined
    if feature_names is None:
        feature_names = ['X'+str(feature+1) for feature in range(features.shape[1])]
    
    print('Fitting linear regression')
    # Multi-threading if the dataset is a size where doing so is beneficial
    if features.shape[0] < 100000:
        model = LinearRegression(n_jobs=-1)
    else:
        model = LinearRegression()
        
    model.fit(features, label)
    
    # Returning linear regression R^2 and coefficients before performing diagnostics
    r2 = model.score(features, label)
    print()
    print('R^2:', r2, '\n')
    print('Coefficients')
    print('-------------------------------------')
    print('Intercept:', model.intercept_)
    
    for feature in range(len(model.coef_)):
        print('{0}: {1}'.format(feature_names[feature], model.coef_[feature]))

    print('\nPerforming linear regression assumption testing')
    
    # Creating predictions and calculating residuals for assumption tests
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    
    def linear_assumption():
        """
        Linearity: Assumes there is a linear relationship between the predictors and
                   the response variable. If not, either a polynomial term or another
                   algorithm should be used.
        """
        print('\n=======================================================================================')
        print('Assumption 1: Linear Relationship between the Target and the Features')
        
        print('Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.')
        
        # Plotting the actual vs predicted values
        sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, height=7)
        
        # Plotting the diagonal line
        line_coords = np.linspace(df_results.min().min(), df_results.max().max(),100)
        plt.plot(line_coords, line_coords,  # X and y points
                 color='darkorange', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.show()
        print('If non-linearity is apparent, consider adding a polynomial term')
        


        
    def normal_errors_assumption(p_value_thresh=0.05):
        """
        Normality: Assumes that the error terms are normally distributed. If they are not,
        nonlinear transformations of variables may solve this.
               
        This assumption being violated primarily causes issues with the confidence intervals
        """
        from statsmodels.stats.diagnostic import normal_ad
        print('\n=======================================================================================')
        print('Assumption 2: The error terms are normally distributed')
        print()
    
        print('Using the Anderson-Darling test for normal distribution')

        # Performing the test on the residuals
        p_value = normal_ad(df_results['Residuals'])[1]
        print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
        # Reporting the normality of the residuals
        if p_value < p_value_thresh:
            print('Residuals are not normally distributed')
        else:
            print('Residuals are normally distributed')
    
        # Plotting the residuals distribution
        plt.subplots(figsize=(12, 6))
        plt.title('Distribution of Residuals')
        sns.distplot(df_results['Residuals'])
        plt.show()
    
        print()
        if p_value > p_value_thresh:
            print('Assumption satisfied')
        else:
            print('Assumption not satisfied')
            print()
            print('Confidence intervals will likely be affected')
            print('Try performing nonlinear transformations on variables')
        
        
    def multicollinearity_assumption():
        """
        Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                           correlation among the predictors, then either remove prepdictors with high
                           Variance Inflation Factor (VIF) values or perform dimensionality reduction
                           
                           This assumption being violated causes issues with interpretability of the 
                           coefficients and the standard errors of the coefficients.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        print('\n=======================================================================================')
        print('Assumption 3: Little to no multicollinearity among predictors')
        
        # Plotting the heatmap
        plt.figure(figsize = (10,8))
        sns.heatmap(pd.DataFrame(features, columns=feature_names).corr(), annot=True)
        plt.title('Correlation of Variables')
        plt.show()
        
        print('Variance Inflation Factors (VIF)')
        print('> 10: An indication that multicollinearity may be present')
        print('> 100: Certain multicollinearity among the variables')
        print('-------------------------------------')
       
        # Gathering the VIF for each variable
        VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
        for idx, vif in enumerate(VIF):
            print('{0}: {1}'.format(feature_names[idx], vif))
        
        # Gathering and printing total cases of possible or definite multicollinearity
        possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
        definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
        print()
        print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
        print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
        print()

        if definite_multicollinearity == 0:
            if possible_multicollinearity == 0:
                print('Assumption satisfied')
            else:
                print('Assumption possibly satisfied')
                print()
                print('Coefficient interpretability may be problematic')
                print('Consider removing variables with a high Variance Inflation Factor (VIF)')
        else:
            print('Assumption not satisfied')
            print()
            print('Coefficient interpretability will be problematic')
            print('Consider removing variables with a high Variance Inflation Factor (VIF)')
        
        
    def autocorrelation_assumption():
        """
        Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                         autocorrelation, then there is a pattern that is not explained due to
                         the current value being dependent on the previous value.
                         This may be resolved by adding a lag variable of either the dependent
                         variable or some of the predictors.
        """
        from statsmodels.stats.stattools import durbin_watson
        print('\n=======================================================================================')
        print('Assumption 4: No Autocorrelation')
        print('\nPerforming Durbin-Watson Test')
        print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
        print('0 to 2< is positive autocorrelation')
        print('>2 to 4 is negative autocorrelation')
        print('-------------------------------------')
        durbinWatson = durbin_watson(df_results['Residuals'])
        print('Durbin-Watson:', durbinWatson)
        if durbinWatson < 1.5:
            print('Signs of positive autocorrelation', '\n')
            print('Assumption not satisfied', '\n')
            print('Consider adding lag variables')
        elif durbinWatson > 2.5:
            print('Signs of negative autocorrelation', '\n')
            print('Assumption not satisfied', '\n')
            print('Consider adding lag variables')
        else:
            print('Little to no autocorrelation', '\n')
            print('Assumption satisfied')

            
    def homoscedasticity_assumption():
        """
        Homoscedasticity: Assumes that the errors exhibit constant variance
        """
        print('\n=======================================================================================')
        print('Assumption 5: Homoscedasticity of Error Terms')
        print('Residuals should have relative constant variance')
        
        # Plotting the residuals
        plt.subplots(figsize=(12, 6))
        ax = plt.subplot(111)  # To remove spines
        plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
        plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine
        plt.title('Residuals')
        plt.show() 
        print('If heteroscedasticity is apparent, confidence intervals and predictions will be affected')
    linear_assumption()
    normal_errors_assumption()
    multicollinearity_assumption()
    autocorrelation_assumption()
    homoscedasticity_assumption()
#%%        

#\"""matching total feeding time with slurry density if  flow is almost constant"""
#fed_in=[]
#T_alim=[]
#density=[]
#fed_fin=[]
#t_s,t_e=select_cycle2_time(dfSlow,'analogSlow21')
#indices_slow=select_cycles_indices_by_time(dfSlow,t_s,t_e)
#indices_fast=select_cycles_indices_by_time(dfFast,t_s,t_e)
#indices_slow=indices_slow[1:]
#indices_fast=indices_fast[1:]
#for index in indices_slow:
#    a=np.max(np.array(dfSlow['analogSlow19'])[index[0]:index[1]])
#    fed_in.append(a)
#for index in indices_fast:
#    a= final_feeding_delivery_on_pressure(dfFast,index,14)
#    fed_fin.append(a)  
##    c=np.max(np.array(dfSlow['analogSlow20'])[index[0]:index[1]])
##    density.append(c)
#for i in range(0,len(indices_slow)):
#    if (fed_in[i]>200. and fed_in[i]<260. and fed_fin[i]>0. and fed_fin[i]<10.):
#        b=np.max(np.array(dfSlow['analogSlow21'])[indices_slow[i][0]:indices_slow[i][1]])
#        c=np.max(np.array(dfSlow['analogSlow20'])[indices_slow[i][0]:indices_slow[i][1]])
#        T_alim.append(b)
#        density.append(c)
#plt.scatter(T_alim,density)
#%%
#
#import scipy
#lin_reg=scipy.stats.linregress(T_alim,density)
#xn=np.linspace(min(T_alim),max(T_alim),200)
#yn = lin_reg.intercept+lin_reg.slope*xn
#
#plt.plot(T_alim,density, 'or')
#plt.plot(xn,yn)
#plt.plot(xn,yn+lin_reg.stderr)
#plt.plot(xn,yn-lin_reg.stderr)
#plt.show()

#%%

#"""we want to compare some measure of different cycle
#this function, giving the array of values, theindices of the cycles and the measure we want, give a statistic about the cycles
#"""
#
#def cycle_stat_measure(arr, indices,statistic,statistic_libr):
#    """indices should be a matrix Nx2 or a array of array 2x1"""
#    vals=[]
#    module = __import__(statistic_libr)
#    method_to_call = getattr(module, statistic)
#    for index in indices:
#        vals.append(method_to_call(arr[index[0]:index[1]]))
#    return vals
#
#def boxplot(arr, indices):
#    plt.figure()
#    data=[]
#    for index in indices:
#        data.append(arr[index[0]:index[1]])
#    plt.boxplot(data)


#%%

#%%
def plot_with_same_sampling(df,namevars,path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/immagini generate"):
    for name in namevars:
        for name2 in namevars:
            if name!=name2:
                plt.figure()
                plt.scatter(df[name],df[name2],marker='.')
                plt.xlabel(name)
                plt.ylabel(name2)
                plt.title(name+'-'+name2)
                plt.savefig(path+"/"+name+"-"+name2)
                plt.close()
                
                
def plot_with_different_sampling(df1,df2,var1,var2,timename='Time number',
                                     path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/immagini generate"):
    time1=df1['Time number']
    time2=df2['Time number']
    for name1 in var1:
        for name2 in var2:
            var2_interp=np.interp(time1,time2,var2)
            fig, axs = plt.subplots(2)
            axs[0]=plt.scatter(time1,var2_interp,marker='.')
            axs[0]=plt.scatter(time2,var2,marker='o',alpha=0.2)
            axs[1]=plt.scatter(var2_interp,var1)


                


"""



#%%
df_analog1_3=pd.DataFrame()
for file in phase2:
    df,df_phase=df_from_phase_bson(file,path1)
    df=add_time_as_timeseries(df,'timestamp')
    df=df[['timeserie','Analogs.analog1','Analogs.analog3']]
    df_analog1_3=pd.concat([df_analog1_3,df],axis=0)
for file in phase3:
    df,df_phase=df_from_phase_bson(file,path1)
    df=add_time_as_timeseries(df,'timestamp')
    df=df[['timeserie','Analogs.analog1','Analogs.analog3']]
    df_analog1_3=pd.concat([df_analog1_3,df],axis=0)

df_analog1_3= df_analog1_3.set_index('timeserie')   
df_analog1_3= df_analog1_3.sort_index()

#%%
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# Import Data
df=df_analog1_3
# Multiplicative Decomposition 
#result_mul = seasonal_decompose(df['Analogs.analog1'], model='multiplicative', extrapolate_trend='freq')

# Additive Decomposition
result_add = seasonal_decompose(df['Analogs.analog1'], model='additive', )

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plt.show()
"""
