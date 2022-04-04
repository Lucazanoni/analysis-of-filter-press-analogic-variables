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
from datetime import datetime
path1="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/15 settembre- 4 ottobre dati/bson"
path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro"
pathprova="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/risultati/immagini_prova"
path2="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/15 settembre- 4 ottobre dati/power 15 sett - 9 novembre"
path3="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/15 settembre- 4 ottobre dati/prova"
pathEventLog="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/15 settembre- 4 ottobre dati/energy 15 sett - 9 novembre/CycleEventLog"
pathCycleLog="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/15 settembre- 4 ottobre dati/energy 15 sett - 9 novembre/CycleLog"
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

analogflow='Analogs.analog1'
analogpressure='Analogs.analog3'
analogdensity='Analogs.analog22'
"""cicli esclusi perchè solo relativi a fasi non di pompaggio + iniziali dopo almeno 2 giorni di stop"""
#cicli_da_escludere=[70,72,74,76,78,81,85,94,108,119,147,148,149,155,156,157,164,227,230,235,246,261,262]

"""cicli esclusi perchè  solo relativi a fasi non di pompaggio + """
"""iniziali dopo almeno 2 giorni di stop + cicli che durano > 6h"""
cicli_da_escludere=[70,72,74,76,78,81,85,94,108,119,144,146,147,148,149,155,156,157,160, 164,184,192,200,201,226,227,
                    229,230,234,235,239,240,246,249,257,261,262]

#%%

def change_global_names(density='Analogs.analog22',flow='Analogs.analog1',pressure='Analogs.analog3'):
    """
    This function create 3 global variables (analogflow,analogpressure,analogdensity) with the name of the variables 
    used in the dataframe.  
    Parameters:
        density: string with the name of the pressure in the Dataframe. Default: 'Analogs.analog22'
        flow: string with the name of the flow in the Dataframe. Default: 'Analogs.analog1'
        pressure: string with the name of the pressure in the Dataframe. Default: 'Analogs.analog3'
    Return:
        Create (or replace) 3 global variables called analogflow, analogpressure and analogdensity that are strings
    """
    
    analogflow='analogflow'
    analogpressure='analogpressure'
    analogdensity='analogdensity'
    globals()[analogflow]=flow
    globals()[analogdensity]=density
    globals()[analogpressure]=pressure



#%%

def extract_bson_files_from_zip(zip_path,destination_path):
    """ EXTRACT ZIP FILES
        
        if bson files are zipped and the contain a folder named output that contain the bson file
        getting all filename of zipfiles in the zip_path directory""" 
    import zipfile
    files=[]
    os.chdir(zip_path)
    for file in glob.glob("*.zip"):
        files.append(file)
    """extract files in the target directory"""
    for file in files:
        with zipfile.ZipFile(file,"r") as zip_ref:
            zip_ref.extractall(destination_path)

#%%
"""READING AND OPENING FILES JSON AND BSON"""


def read_json_names(path):
    
    """
    Read the name of json files in a given directory
    Parameter:
        path: string of path of a given directory
    return:
        list of strings of json files in directory""" 
    files=[]
    os.chdir(path)
    for file in glob.glob("*.json"):
        files.append(file)
    return files

def json_file_to_df(filenames, path):
    
    """
    Create a global Pandas DataFrame for each json file read
    
    Parameters:
        filenames: list of strings of json files
        path: string of the path of a given directory
    returns:
        creates global pandas DataFrames of the json files names present in filenames exixting in directory selected
    
    """
    os.chdir(path)
    for file in filenames:
        file1=file[33:-5]
        file1=file1.replace(" ","")
        file1=file1.replace("(","_")
        file1=file1.replace(")","")
        globals()[file1]=pd.read_json(file)
        
#def read_json_files(name,path):
#    """
#    Read the json files in a directory
#    
#    Parameters:
#        name: string conteining the json file name
#        path: string of the path of a given directory
#        
#    return
#    """
#    with open(path+'/'+name) as json_file:
#        return json.load(json_file)


def make_bson_list_for_phase(path,numbers_of_phases):
#example:  make_bson_list_for_phase(path_prova,[1,2,3])
    """
    Create a global list of strings for each phase of interst with the name of all the bson files of that phase.
    The phase of each file should be in the name of the file. Ex: 'AQS_cycle20_phase5.bson', where the phase is 'phase5' in this case

    
    Parameters:
        path: string of the path of a given directory
        number of phase: array containing the phases of interest. Ex: [1,2,3,5]
    Return: 
        a global list with the names (strings) of all bson files in selected directory for each phase of interest.   
    """
    files=[]
    os.chdir(path)
    for file in glob.glob("*.bson"):
        files.append(file)
    for i in numbers_of_phases:
        names=[]
        for file in files:
            if ('phase'+str(i)) in file:
                names.append(file)
        globals()['phase'+str(i)]=names



def df_from_bson(filename,path):
    """
    Create a Pandas DataFrame from bson file
    
    Parameters:
        filename: name (string) of the bson file 
        path: string of the path of a given directory
    return:
        df: Pandas DataFrame of the starting bson file
    
    """    
    with open(path+'/'+filename, "rb") as rf:
        data = bson.decode(rf.read())
    df=pd.DataFrame(data[list(data)[0]])
    df.dropna(subset = ['timestamp'], inplace=True)
    return df



#generate a dataframe pandas from bson file for phase variable, in whing there are 2 db, one with analogs and another with phase
def df_from_phase_bson(filename, path):
    """
    Create 2 Pandas DataFrames from bson file of phase variable. The phase bson files provided has 2 different dataframe in same file,
    so it is necessary to separe them for the differen use of the variables
    
    Parameters:
        filename: name (string) of the bson file 
        path: string of the path of a given directory
    return:
        df: Pandas DataFrame of the "normal/analog" variables of bson file
        df_phase: Pandas DataFrame of the "phase" variables of bson file
    
    """   
    
    
    with open(path+'/'+filename, "rb") as rf:
        data = bson.decode(rf.read())
    df=pd.DataFrame(data[list(data)[0]])
    df.dropna(subset = ['timestamp'], inplace=True)
    df_phase=pd.DataFrame(data[list(data)[1]])
    df_phase.dropna(subset = ['timestamp'], inplace=True)
    return df,df_phase                
   
    

def cycle_list_file(n_cycle,path):
    """
    Create a list of names of all file in a certain directory that has a certain number in position 9-10.
    For example:  'AQS_cycle62_phase2.bson' has number 62
    Useful for selecting the cycles of interest
    
    Parameters:
        n_cycle: int of number of cycle. It must be a number between 10 and 99 ( 2 digit)
        path:string of the path of the directory where the files are
    
    Returns: 
        names: list of strings of the files with 'n_cycle' in position 9-10 of the name in the selected directory    
    
    
    """
    files=[]
    os.chdir(path)
    for file in glob.glob("*.bson"):
        files.append(file)
    names=[]
    for file in files:
        if file[9:11]==str(n_cycle):
            names.append(file)
    return names       
#%%            
            
#TIME IN FORM OF NUMBER FROM DATES ISO OR DATETIME FORMAT

def take_datetime(df,timename='_time'):
    """add in Pandas dataframe with time data in iso8601 formata a column with data in datastamp format 
    PARAMETERS:
        df: Pandas dataframe
        timename: the column name of the iso8601 time data. default: '_time'
        
    Returns:
        The initial dataframe with a new column in last position with time in datastamp format, called 'time' 
    
    """
    dates=[]
    for date in df[timename]:
        date=dateutil.parser.isoparse(date[:-1])
        dates.append(date)
    date=pd.DataFrame({'time':dates})
    return date


def timefromiso(dataiso):
    """function to elaborate iso8601 time format, useful for other functions"""
    return datetime.fromisoformat(dataiso[:-1])

#trasforma orario formato hh,mm,ss,micros in secondi 
def time_to_num(t):
    """return the number of seconds in TimeStamp object
    Parameter:
        t: TimeStamp object 
    Return:
        number of seconds in t as float
    """ 
    return t.timestamp()#+t.microseconds*10**(-6)
    
#add to df a column of time in second if time column is in iso format
def add_time_as_number(df,timename='_time'):
    """add in Pandas dataframe with time data in iso8601 formata a column with data in seconds 
    PARAMETERS:
        df: Pandas dataframe
        timename: the column name of the iso8601 time data. default: '_time'
        
    Returns:
        The initial dataframe with a new column in last position with time in seconds, called 'time', 
        where the first value is the time starting point (0.0s) 
    Raise: 
        ValueError if the called column does not exist in dataframe
    
    """
    if not(timename in df.columns):
        raise ValueError(timename+' do not exist in the dataframe')
    timenumberdf=pd.DataFrame({"Time number":numbers_from_time(df,timename)})
    return(pd.concat([df,timenumberdf],axis=1))
    
#add to df a column of time in second if time column is in datetime format
def add_time_as_number2(df,timename='_time'):
    """
    Add in Pandas DataFrame with time data in timestamp formata a column with data in seconds 
    Parameters:
        df: Pandas dataframe
        timename: the column name of the timestamp time data. default: '_time'
        
    Returns:
        The initial dataframe with a new column in last position with time in seconds, called 'time', 
        where the first value is the time starting point (0.0s) 
    Raise: 
        ValueError if the called column does not exist in dataframe
    
    """
    if not(timename in df.columns):
        raise ValueError(timename+' do not exist in the dataframe')
    timenum=[]
    t0=time_to_num(df[timename].iloc[0])
    for time in df[timename]:
        timenum.append(time_to_num(time)-t0)
    timenumberdf=pd.DataFrame({"Time number":timenum})
    return pd.concat([df,timenumberdf],axis=1)


def numbers_from_time(df,timename='_time'):
    """
    Transfom the column of Dataframe referred to time in timestamp format in seconds
    Parameters:
        df: the Pandas DataFrame which has the time column to transform in seconds
        timename: string of the name of the time column. Default: '_time' 
    Returns:
        array of float containing the element of df[timename] transformed in seconds 
    Raise: 
        VauleError if df[timename] does not exist
    """
    timenum=[]
    if not(timename in df.columns):
        raise ValueError(timename+' do not exist in the dataframe')
    
    t0=time_to_num(timefromiso(df[timename][0]))
    for time in df[timename]:
        t=timefromiso(time)
        timenum.append(time_to_num(t)-t0)
    return (np.array(timenum))

def add_time_as_timeseries(df,timename='timestamp'):
    """
    Add in Pandas DataFrame with time data in iso8601 a column with data in timestamp format
    Parameters:
        df: Pandas dataframe which has the time column in iso8601
        timename: the column name of the timestamp time data. default: 'timestamp'
        
    Returns:
        The initial dataframe with a new column in last position with time in seconds, called 'timeserie', 
        where the first value is the time starting point (0.0s) 

    """
    timeserie=[]
    for time in df[timename]:
        timeserie.append(timefromiso(time))
    timeserie=pd.DataFrame({'timeserie':timeserie})
    return pd.concat([df,timeserie],axis=1)
#%%            
#take the max of a certain phase of each cycle for each of the variable of interest
#def max_of_phase(files, path,variables):
#    
#    x=np.zeros([len(files),len(variables)])
#    time=[]
#    j=0
#    for file in files:
#        x1=np.zeros(len(variables))
#        df,df_phase=df_from_phase_bson(file, path)
#        i=0
#        for var in variables:
#            x1[i]=max(np.array(df_phase[var]))
#            i=i+1
#        x[j,:]=x1
#        df_phase=add_time_as_timeseries(df_phase)
#        time.append(df_phase['timeserie'][len(df_phase)-1])
#        j=j+1
#    sorted_index = np.argsort(time)
#    x1=np.zeros([len(files),len(variables)])
#    time = [time[i] for i in sorted_index]
#    j=0
#    for i in sorted_index:
#        x1[j,:]=x[i,:]
#        j=j+1
#    return x1,time
#
#


#%%
    



def t_V_over_V_slopes(time,volume,starting_points=200,slope_points=20,limit=2):
    """
    This function find if there is a changing in the slope of time/volume over volume plot, that should be linear.
    When it's not linear and there is a changing in the slope, the 2 parts of the curve are separated in the linear and the second part.
    The assumption is that the initial part is linear and then calculate when the slopes diverge fron the initial one.
    
    Parameters:
        time: array of float containing the times at which each measure of slurry volume is collected
        volume: array of float of the measured volume of slurry
        starting_points: number (integer) of points used to determine the starting slope of the curve. the index of first point is always 0, 
                         so the firsts starting_points points will be used for the calculation of the slope of the linear part of the curve.
                         Default=200
        slope_points: integer number fo points used to evaluate the slope of the curve after the first starting_points points. The slope is
                      not the tanget of the 'time/volume over volume' curve but a mean to avoid that fluctuations can misrepresent the result.
                      Default=20
        limit: float of limit ratio between the starting slope and the slope calculated in succesive points. if the ratio is bigger than limit,
               the point in which that happens is consider the changing slope point. 
               Default=2
    
    Returns:
        if there is a change in the slope of 'time/volume over volume' curve, the index point in which that happens is returned. Otherwise,
        it is returned 0.
    
    
    """
    t_v=np.array(time/volume)[1:]
    volume=volume[1:]
    #i calculate the starting slope of the curve assuming is linear
    start_slope=sp.stats.linregress(volume[:starting_points],t_v[:starting_points])[0]  
    for i in range(200,len(t_v)-20):
        #i calculate each point the slope of the next 20 points to avoid fluctuation
        slope=sp.stats.linregress(volume[i:i+slope_points],t_v[i:i+slope_points])[0] 
        
        if slope/start_slope>limit:                                  
            # if the slope of next points are over the double of starting slopes, i break the curve 
            #print("the changing of slope occours in position",i)
            return i
    #print("no change of slope")
    return 0 
    #if there is no significative change of slope return 0

#%%
#i want that the derivate of pressure is null for a while, so i check if the next n=20 points have derivate very little (<0.01)
def limit_pressure(pressure,time,p_range=.5,n_limit=20,figure=False):
    """
    This function finds the pressure value after which the pressure can be considered almost constant, considering fluctuations, 
    after reaching the flow rate desired. It uses the derivative of the pressure and consider when it's almost  null (<0.01)
    for a certain number of point.
    It can also show the point find plotting the graph of pressure and remarking the point found with a red dot
    
    Parameters:
        pressure: array or list of float, conteining the sequence of measured pressures 
        time: array or list of float, containing the times at which each pressure point is register. 
              len(pressure) should be equal to len(time)
        p_range: float. the distance from maximum value of pressure in which i search a region of pressure constant.
                 This is done because in some points of the curve there can be some region 
                 of constant pressure before arriving to the desired range of pressure, that should be near the maximum.
                 Default=0.5
        n_limit: integer number of consecutive points that should have simultaneusly derivative <0.01 to define the pressure is constant
                 Default=20
        figure: boolean. True to plot pressure and point of starting constant pressure, False otherwise
                Default=False
              
    
    """
    derivate=np.gradient(pressure,time)
    counter=False
    for i in range(len(pressure)-n_limit):
        if (abs(derivate[i])<0.01 and abs(max(pressure)-pressure[i])<p_range):
            counter=True
            for j in range(i,i+n_limit):
                if abs(derivate[j])>0.01 and abs(max(pressure)-pressure[i])<p_range:
                    counter=False
            if counter:
                if figure:
                    plt.figure()
                    plt.scatter(time,pressure,marker='.')
                    plt.scatter(time[i],pressure[i],color='red')
                    from datetime import datetime
                    n= datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                    plt.savefig(pathprova+'/'+n+'.png')
                    plt.close()
                return i
    print("pressure not constant")
    return 0
#%%

   #da fare
def time_volume_over_volume(filenames, path,slope_limit=2,starting_points=200,slope_points=20,figure=False):
    """
    This function calculate the parameters of the linear regression of time/volume- volume curve in the linear part (phase 3)
    It does t for all the target files in a directory and join the result
    in theory this relation should be linear if the pressure is constant, and after few seconds in phase 3 of the cycles it is true
    if there is a change in the slope the function the changing point for each cycle.
    It can also 
    
   
   
   
   
    """
    slopes=[]
    intercepts=[]
    err=[]
    r_values=[]
    times=[]
    indices=[]
    """index is where the curve is considered no more linear. if zero it is considered linear in all the region at constant pressure"""
    i=0
    """i is the number for the figure"""
    for file in filenames:
        i=i+1
        df,df_phase=df_from_phase_bson(file, path)
        df=add_time_as_number(df,'timestamp')
        limit=max(np.array(df[analogpressure]))-1 #limit over which the pressure remain constant
        res = next(x for x, val in enumerate(df[analogpressure]) if val > limit)
#        res=limit_pressure(df['Analogs.analog3'],df['Time number'],n_limit=30)
#        if res==0:
#            print(file+" don't have a sufficiently constant pressure" )
#            continue
#        
        #res2 = next(x for x, val in enumerate(df[pressure][res:]) if val <limit-3 ) #when pressure drops in the last seconds it cannot be more considered constant
        res2=-5
        volume=volume_from_flow(np.array(df[analogflow])[res:res2],np.array(df['Time number'])[res:res2])
        time=np.array(df['Time number'])[res:res2]-df['Time number'][res]
        #calculate T over V in the region of costant pressure 
        t_V=(df['Time number'][res:res2]-df['Time number'][res])/volume
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(volume[1:],t_V[1:])
        index=t_V_over_V_slopes(time,volume,slope_points=slope_points,limit=slope_limit,starting_points=starting_points)   #find the index in which the curve is not linear anymore, if exist   
        indices.append(index)
        
        if figure: 
            if index!=0:
                plt.figure()
                plt.scatter(volume[:index],t_V[:index],marker='.')#c=df['Time number'][res:res+i])
                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(volume[1:index],t_V[1:index])
                x=np.linspace(min(volume[:index]),max(volume[:index]),100)
                plt.plot(x,intercept+slope*x,linestyle='--',color='red')
                plt.scatter(volume[index:],t_V[index:],marker='.')#c=df['Time number'][res+i:res2])
#                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(volume[index:],t_V[index:])
#                x=np.linspace(volume[index],volume[-1],100)
#                plt.plot(x,intercept+slope*x,linestyle='--',color='green')
                
                
            else:
                plt.figure()
                plt.scatter(volume,t_V,marker='.',c=df['Time number'][res:res2])
                x=np.linspace(min(volume),max(volume),100)
                plt.plot(x,intercept+slope*x,linestyle='--')
            #plt.colorbar()
            #plt.title('final liquid concentration='+ str(liquid_concentration[-1]))
            
            #print('r_value = ',r_value)
#            x=np.linspace(min(volume),max(volume),100)
#            plt.plot(x,intercept+slope*x,linestyle='--')
            plt.xlabel('V   m3')
            plt.ylabel('t/V     s m^(-3)')
            
            
        slopes.append(slope)
        intercepts.append(intercept)
        err.append(std_err)
        r_values.append(r_value)
        times.append(timefromiso(df['timestamp'][res]))
        #date.append(df['timestamp'][res])
    return slopes,intercepts,err,r_values,times,indices
#%%

def slopes_time_volume_over_volume(flows,times):
    """
    This function calculate the derivative of the curve time/volume-volume starting from instant flows and time.
    It's used to find the change of slope in the curve
    
    Parameters:
        flows: array of double containing the the measured flows
        time: array of double containing the times (in seconds) at which each flow is measured. it must be that len(flows)==len(times)
    
    Return:
        array of double containing the gradient of the curve time/volume-volume
    """
    
    times=times-times[0]
    volume=volume_from_flow(flows,times)
    t_V=np.array(times[1:])/np.array(volume[1:])
    return np.gradient(t_V,volume[1:])



#%%
#def specific_cake_resistances(filenames,path,solid_density=2.65746,liquid_density=1.):
#    """
#    This function calculate the specific resistance of the cake during his formation in phase 3.  
#    
#    
#    
#    alpha=[] 
#    #the specific cake resistance per m2 when the pressure is almost constant"""
#    x=time_volume_over_volume(filenames,path)
#    err=x[2]
#    slopes=x[0]
#    rel_slope_err=np.array(err)/np.array(slopes) 
#    """relative error of slopes"""
#    solid_con=solid_concentrations( filenames,path)
#    mean_pressure=[]
#    err_pressure=[]
#    for file in filenames:        
#        df,df_phase=df_from_phase_bson(file,path)
#        limit=max(np.array(df_phase['PhaseVars.phaseVariable4']))-1
#        res = next(x for x, val in enumerate(df[analogpressure]) if val > limit)
#        res2 = next(x for x, val in enumerate(df[analogpressure][res:]) if val <limit-2 )
#        mean_pressure.append(np.mean(np.array(df[analogpressure])[res:res2]))
#        err_pressure.append(np.std(np.array(df[analogpressure])[res:res2]))
#    rel_press_err=np.array(err_pressure)/np.array(mean_pressure) 
#    """relative error of pressure"""
#    rel_err=rel_slope_err+rel_press_err
#    alpha=np.array(slopes)*np.array(mean_pressure)/np.array(solid_con)
#    alpha_err=rel_err*alpha
#    return alpha,alpha_err
#
#

#%%
    
#here i calculate the residual humidity, that is the mass of H2O residual in the cake
#i have only solid density, slurry density, liquid density and slurry volume


def residual_humidity_over_time(flow,time,slurry_density,figure=False,liquid_density=1.,final_volume=10.8768,solid_density=2.65746):
    """
    This function calculate the residual humidity, i.e. the residual mass of water in the cake, during the formation of the cake.
    Parameter:
        flow: array of double containing the instant flows
        time: array of double containing the times (in seconds) at which each flow is measured. it must be that len(flows)==len(times)
        slurry_density: array of double containing the instant densities of the slurry or double containing the mean value
        figure: boolean, True for plotting the changing of residual humidity during the time, False otherwise. Default: False
        liquid_density: density of water extracted. In general it's approximate to 1. Default: 1.
        final_volume: volume of the chamber in which the slurry is pumperd. it's the volume of the cake after the filtration. Default:10.8768
        solid_density: approximate mean density of the solid part of the slurry. Default:2.65746
        
    Return: 
        Array of double containing the residual humidity from the time in which the slurry volume pumped is major than final_volume. the
        lenght of this array is variable. 
        if Figure==True, it will also plotted the array in function of time
    """
    
    
    
    pumped_volume=np.array(volume_from_flow(flow,time))
    solid_concentration=solid_density*(liquid_density-slurry_density)/(slurry_density*(liquid_density-solid_density))
    num=(1-solid_concentration)*(pumped_volume*slurry_density)-(pumped_volume-final_volume)*liquid_density
    den=(pumped_volume*slurry_density-liquid_density*(pumped_volume-final_volume))
    liquid_concentration= num/den
    index=next(x for x, val in enumerate(pumped_volume) if val > final_volume)
    if figure:
        plt.scatter(time[index+1:],liquid_concentration[index+1:],marker='.')
    return liquid_concentration



   
def final_residual_humidity(pumped_volume,slurry_density,liquid_density=1.,final_volume=10.8768,solid_density=2.65746):
    """
    This function calculate the residual humidity, i.e. the residual mass of water in the cake, at the end of the the process.
    Parameters:
        pumperd_volume: double, the total volume of the slurry pumped
        slurry_density: double containing the mean value of the density of slurry
        liquid_density: density of water extracted. In general it's approximate to 1. Default: 1.
        final_volume: volume of the chamber in which the slurry is pumperd. it's the volume of the cake after the filtration. Default:10.8768
        solid_density: approximate mean density of the solid part of the slurry. Default:2.65746
    
    Return: double,  the value of residual humidity at the end of the process
    
    """
    
    solid_concentration=solid_density*(liquid_density-slurry_density)/(slurry_density*(liquid_density-solid_density))
    return (((1-solid_concentration)*pumped_volume*slurry_density-(pumped_volume-final_volume)*liquid_density)/(pumped_volume*slurry_density
           -liquid_density*(pumped_volume-final_volume)))

#density of the cake
def cake_density_over_time(slurry_density,flow,time,liquid_density=1.,final_volume=10.8768,solid_density=2.65746):
    """
    This function calculate the density of the cake during his formation.
    Parameters:
        slurry_density: array of double containing the instant densities of the slurry or double containing the mean value
        flow: array of double containing the instant flows
        time: array of double containing the times (in seconds) at which each flow is measured. it must be that len(flows)==len(times)
        liquid_density: density of water extracted. In general it's approximate to 1. Default: 1.
        final_volume: volume of the chamber in which the slurry is pumperd. it's the volume of the cake after the filtration. Default:10.8768
        solid_density: approximate mean density of the solid part of the slurry. Default:2.65746
    
    Return: array of double containing the density of the cake during his formation, from time[1] to his final value 
    """
    solid_concentration=solid_density*(liquid_density-slurry_density)/(slurry_density*(liquid_density-solid_density))
    pumped_volume=volume_from_flow(flow,time)

    cake_density=np.array(pumped_volume)*solid_concentration*slurry_density/final_volume
    return cake_density
#%%
#def solid_concentrations(filenames,path,solid_density=2.65746,liquid_density=1.):
#    solid_concentrations=[]
#    for file in filenames:
#        df,df_phase=df_from_phase_bson(file,path)
#        slurry_density=np.mean(np.array(df[analogdensity]))
#        solid_concentration=solid_density*(liquid_density-slurry_density)/(slurry_density*(liquid_density-solid_density))
#        solid_concentrations.append(solid_concentration)
#    return solid_concentrations
#%%
#
#def all_final_humidity(filenames,path):
#    liquid_concentrations=[]
#    for file in filenames:
#        df,df_phase=df_from_phase_bson(file,path)
#        liquid_concentrations.append(final_residual_humidity(np.array(df_phase['PhaseVars.phaseVariable6'])[-1],
#                                                                        np.mean(np.array(df_phase['PhaseVars.phaseVariable9']))))
#    return liquid_concentrations
    #%%
#we should mean the slurry density
"""NON SO SE METTERLA"""   
    
    
    
    
    
def all_final_residual_humidity(phase2,phase3,path1, cycle_name=False,errorbar=False,figure=False):
    """
    This function calculate the residual humidity, i.e. the residual mass of water in the cake, at the end of a cycle
    for all the cycles selected and can plot them with their error. It requires the phase 2 and phase 3 of each cycle  
    with all the measured variables. It can also plot the results with or without the errorbar.
    
    Parameters:
        phase2: list of names (as strings) of the files corrisponding to phase 2 of the cycles.  
        phase3: list of names (as strings) of the files corrisponding to phase 3 of the cycles. 
                length of phase3 must be equal to the one of phase2 and files of same cycle should be at same index in phase2 and phase3
        path1: string with path of directory where the files are
        cycle_name: 
    
    
    
    """
    
    
    final_liquid_concentration=[]
    cycle=[]
    error_bar=[]
    for i in range(len(phase3)):
        df2,df_phase2=df_from_phase_bson(phase2[i],path1)
        df3,df_phase3=df_from_phase_bson(phase3[i],path1)
        date=list(add_time_as_timeseries(df2)['timeserie'])
        date.extend(add_time_as_timeseries(df3)['timeserie'])
        flow=np.zeros(len(df2)+len(df3))
        time=np.zeros(len(df2)+len(df3))
        t0=time_to_num(date[0])
        j=0
        for t in date:
            time[j]=time_to_num(t)-t0
            j=j+1
        df2=add_time_as_number(df2,'timestamp')
        df3=add_time_as_number(df3,'timestamp')
        density=np.zeros(len(df2)+len(df3))
        #t2_3=time_to_num(timefromiso(np.array(df3['timestamp'])[0]))-time_to_num(timefromiso(np.array(df2['timestamp'])[-1]))
        flow[:len(df2)]=np.array(df2[analogflow])
        flow[len(df2):]=np.array(df3[analogflow])
        density[:len(df2)]=np.array(df2[analogdensity])
        density[len(df2):]=np.array(df3[analogdensity])
        err_density=np.std(density)
        density=np.mean(density)

        if figure:
            plt.figure()
        err_volume=volume_error(flow,time)
        volume=volume_from_flow(flow,time)[-1]
        x=residual_humidity_over_time(flow,time,density,figure)
        final_liquid_concentration.append(x[-1])
        if cycle_name:
            cycle.append(i[:-24])
        if errorbar:
            err=residual_humidity_error(density,err_density,volume,err_volume)
            error_bar.append(err)

    if cycle_name and errorbar:
        return final_liquid_concentration,error_bar, cycle
    if cycle_name:
        return final_liquid_concentration,cycle
    if errorbar:
        return final_liquid_concentration,error_bar
    return final_liquid_concentration


#%%
#"""i want to correlate the slope of the curve time-volume over volume with the residual humidity"""
#"""non correttissima sulla creazione delle figure, da correggere se mi servirà"""
    

"""QUESTA LA METTA A COOMENTO PERO' POTREI GUARDARCI"""


#def plot_tV_V_slope_and_humidity(df_phase2,df_phase3,plot_figure=False,savefigure=False):
##df_phase2 is the df of analogs in phase2, df_phase3 is the df of analogs in phase 3"""
#    
#        flow=np.zeros(len(df_phase2)+len(df_phase3))
#        time=np.zeros(len(df_phase2)+len(df_phase3))        
#        date=list(add_time_as_timeseries(df_phase2)['timeserie'])
#        date.extend(add_time_as_timeseries(df_phase3)['timeserie'])
#        flow[:len(df_phase2)]=np.array(df_phase2[analogflow])
#        flow[len(df_phase2):]=np.array(df_phase3[analogflow])
#        t0=time_to_num(date[0])
#        i=0
#        for t in date:
#            time[i]=time_to_num(t)-t0
#            i=i+1
#        density=np.zeros(len(df_phase2)+len(df_phase3))
#        density[:len(df_phase2)]=np.array(df_phase2[analogdensity])
#        density[len(df_phase2):]=np.array(df_phase3[analogdensity])
#        mean_density=np.mean(density)
#        residual_humidity=residual_humidity_over_time(flow,time,mean_density,figure=False)
#        limit=max(np.array(df_phase3[analogpressure]))-.5   #limit after that i consider constant the pressure
#        res = next(x for x, val in enumerate(df_phase3[analogdensity]) if val > limit)
#        
#        #res2 = next(x for x, val in enumerate(df['Analogs.analog3'][res:]) if val <limit-2 )
#        res=res+len(df_phase2)
#        #res2=res2+len(df_phase2)
#        vol=volume_from_flow(flow,time) 
#        vol_index=next(x for x, val in enumerate(vol) if val > 10.8768)  #index which before the pumped volume is less than the filter volume
#                                                                        # and so the formula is not valid
#
#        #density=np.mean(density)
#        solid_concentration=2.65746*(1-mean_density)/(mean_density*(1-2.65746))
#        
#        volume=volume_from_flow(flow[res:-5],time[res:-5])
#        t_V=(time[res+1:-5]-time[res])/volume[1:] #i calculate t/V
#        
#        """ slope of t_V over V in function of humidity"""
#        #the first 200 points in the reagion of constant pressure are used to define the starting slope, 
#        #we then compute the derivate for next slopes
#        
#        slopes=np.gradient(t_V,volume[1:])
#        date1=date[-1].strftime("%Y-%m-%d-%H-%M")
#       
#        if plot_figure:
#            plt.axhline(y=0.2,color='red',linestyle='-.')
#            plt.scatter(slopes,(residual_humidity[res+1:-5]))
#            plt.xlabel('log(slope t_V over V)')
#            plt.ylabel('residual humidity')
#            path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/risultati/immagini_prova"
#            if savefigure:
#                plt.savefig(path+'/slope-hum-'+date1+'.png')
#                plt.close()
#        
#        """alpha and humidity"""
#        
#        mean_pressure=np.mean(np.array(df_phase3['Analogs.analog3'][res:-5]))
#        alpha=slopes*mean_pressure/solid_concentration#[res+1:-5]
#        if plot_figure:
#            plt.figure()
#            plt.scatter(alpha,residual_humidity[res+1:-5])
#            plt.xlabel('alpha')
#            plt.ylabel('residual humidity')
#            plt.axhline(y=0.2,color='red',linestyle='-.')
#            if savefigure:
#                plt.savefig(fname=path+'/alfa-hum-'+date1+'.png')
#                plt.close()
#
#        
#        
#        """plotting t_v over V and residual humidity over time"""
#        if plot_figure:
#
#            fig,ax=plt.subplots(2,1)
#            fig.suptitle("residual humidity = "+str(residual_humidity[-1]),fontsize='10')
#            ax[0].plot_date(date[vol_index:],residual_humidity[vol_index:],marker='.')
#            #ax[0].axvline(x=date[vol_index])
#            ax[1].scatter(volume[1:],t_V,marker='.',c=time[res+1:-5])
#            if savefigure:
#                fig.savefig(fname=path+'/t_V over V-hum- '+date1+'.png')
#                plt.close()  

#%%
                
#density should be mean cause the too big fluctuation of the read values"""
def residual_humidity_of_changing_slope(filenames2,filenames3,path,mean_density=True,solid_density=2.65746,liquid_density=1.):
    """
    This function calculate the residual humidity, i.e. the residual mass of water in the cake, in the point of the curve
    time/volume-volume in which there is the change of slope for all the cycles selected.
    
    Parameters:
        filenames2: list of string of the name of the files of phases 2 of desired cycles.
        filenames3: list of string of the name of the files of phases 3 of desired cycles. the number of elements of 
            filenames3 must be equal to the one of filenames2 and files of same cycle should be at same index in filenames2 and filenames3
        path: string of the path to the directory in which the files are
        mean_density: boolean. True if the slurry density used is the mean of the istantaneous slurry density measured. False to use all 
            the istantaneous density measured are used. Default: True
        solid_density:  solid_density: approximate mean density of the solid component of the slurry. Default:2.65746
        liquid_density: density of water extracted. In general it's approximate to 1. Default: 1.
    
    Returns:
        slope_changing_residual_humidity: array of double containing the residual humidity of each cycle in the point of changing slope
            the index of each element is the same to the one of the corrisponding cycle in filenames2 nd filenames3 
        slope_changing_control: array of integer that check for each cycle if there a point of changing slope. The element is 0 if
            there is no changing of slope in the curve, and the index of changing slope of the corrisponding cycle otherwise. 
            the indices of this array corrispond to the ones in filenames2 and filenames3 (same indices, same cycle) 
    """
    slope_changing_residual_humidity=[]

    slope_changing_control=np.zeros(len(filenames2))
    for i in range(len(filenames2)):
        df2,df_p2=df_from_phase_bson(filenames2[i],path)
        df3,df_p3=df_from_phase_bson(filenames3[i],path)
        flow=np.zeros(len(df2)+len(df3))
        time=np.zeros(len(df2)+len(df3))        
        date=list(add_time_as_timeseries(df2)['timeserie'])
        date.extend(add_time_as_timeseries(df3)['timeserie'])
        flow[:len(df2)]=np.array(df2[analogflow])
        flow[len(df2):]=np.array(df3[analogflow])
        t0=time_to_num(date[0])
        j=0
        for t in date:
            time[j]=time_to_num(t)-t0
            j=j+1
        density=np.zeros(len(df2)+len(df3))
        density[:len(df2)]=np.array(df2[analogdensity])
        density[len(df2):]=np.array(df3[analogdensity])
        if mean_density:
            mean_density=np.mean(density)
            residual_humidity=residual_humidity_over_time(flow,time,mean_density,figure=False)
        else:
            residual_humidity=residual_humidity_over_time(flow,time,density,figure=False)

        limit=max(np.array(df3[analogpressure]))-1   #limit after that i consider constant the pressure
        idx = next(x for x, val in enumerate(df3[analogpressure]) if val > limit)
        idx=idx+len(df2)

        vol=volume_from_flow(flow,time)

        vol_idx=next(x for x, val in enumerate(vol) if val > 10.8768) #index which before the pumped volume is less than the filter volume
                                                                        # and so the formula is not valid
        idx=max([idx,vol_idx])

        index_change=t_V_over_V_slopes(time[idx:],vol[idx:])
        if index_change!=0:
            slope_changing_residual_humidity.append(residual_humidity[index_change+idx])
        else:
            slope_changing_residual_humidity.append(residual_humidity[-1]) #-5 to avoid the values when the pressure is dropped
            slope_changing_control[i]=1
    return slope_changing_residual_humidity, slope_changing_control
#%%
#calculate the error of the residual humidity
def residual_humidity_error(slurry_density,delta_slurry,volume,delta_volume,solid_density=2.65746,
                            delta_solid=.9,liquid_density=1.,filter_volume=10.8768):
    
    """
    This funtion calculate the uncertainty assiociated with the residual humidity, i.e. the residual mass of water in the cake.
    In this case the values of density of water and filtered volume of water are taken without uncertainty.
    
    Parameters:
        slurry_density: double, density of the slurry
        delta_slurry: double, the uncertainty associated with the slurry density
        volume: double, the pumped volume of slurry
        delta_volume: double, the uncertainty associated with the pumped volume of slurry
        solid_density: double, the density of the solid component of the slurry. Default: 2.65746
        delta_solid: double, the uncertainty associated with the solid component of the slurry. Default: .9
        liquid_density: double, the density of the liquid component of the slurry (typically water). Default: 1.
        filter_volume: volume of the chamber in which the slurry is pumperd. it's the volume of the cake after the filtration. Default:10.8768
    
    Returns:
        double, the uncertainty associated to the residual humidity
    """
    
    
    #rh=residual_humidity
    sl_d=slurry_density
    d_sl=delta_slurry
    v=volume
    dv=delta_volume
    so_d=solid_density
    d_so=delta_solid
    ld=liquid_density
    fv=filter_volume
    #d_residual_humidity/d_slurry_density
    dC_dsl=(sl_d*so_d+so_d*(ld-so_d))/(((ld-so_d)*sl_d**2)*(sl_d*v-ld*(v-fv)))+(so_d*(ld-sl_d)*v)/(sl_d*(ld-so_d)*(sl_d*v-ld*(v-fv))**2)

    #d_residual_humidity/d_slurry_volume
    dC_dv=(so_d*(ld-sl_d)/(sl_d*(ld-so_d)))*(sl_d-ld)/((sl_d*v-ld*(v-fv))**(2)) 

    #d_residual_humidity/d_solid_density
    dC_dso=-1*((ld-sl_d)*(ld-so_d)+so_d*(ld-sl_d))/(sl_d*(sl_d*v-ld*(v-fv))*(ld-so_d)**2)

    err=((dC_dsl*d_sl)**2+(dC_dso*d_so)**2+(dC_dv*dv)**2)**0.5
    print('residual humidity error is ',err)
    return err

def volume_error(flow,time,percentual_flow_error=0.005):
    """
    This function calculate the uncertainty associated to the volume calculated as integral of the flow in time.
    In this function the time is taken without uncertainty, because the strumental error on flow is much bigger than the one on time.
    
    Parameters:
        flow: array of double containing the instant flows
        time: array of double containing the times (in seconds) at which each flow is measured. it must be that len(flows)==len(times)
        percentual_flow_error: the percentual uncertainty associated to the measures of flow. Default: 0.005 (0.5%)
    
    Returns:
        double, the uncertainty associated to the pumped volume 
        
    """
    err=percentual_flow_error*flow
    tot_err=0
    
    for i in range(len(flow)-1):
        tot_err=tot_err+(err[i]+err[i+1])*0.5*(time[i+1]-time[i])
    return tot_err




#%%
#"""this function plot the time volume over volume, but showing which point corrispond to the first point that reach a trget humidity"""
#
#def plot_slopes_of_taget_humidity(filenames2,filenames3,path,target_humidity):
#
#    for i in range(len(filenames2)):
#        df2,df_p2=df_from_phase_bson(filenames2[i],path)
#        df3,df_p3=df_from_phase_bson(filenames3[i],path)
#        flow=np.zeros(len(df2)+len(df3))
#        time=np.zeros(len(df2)+len(df3))        
#        date=list(add_time_as_timeseries(df2)['timeserie'])
#        date.extend(add_time_as_timeseries(df3)['timeserie'])
#        flow[:len(df2)]=np.array(df2[analogflow])
#        flow[len(df2):]=np.array(df3[analogflow])
#        t0=time_to_num(date[0])
#        j=0
#        for t in date:
#            time[j]=time_to_num(t)-t0
#            j=j+1
#        limit=max(np.array(df3[analogpressure]))-1   #limit after that i consider constant the pressure
#        idx = next(x for x, val in enumerate(df3[analogpressure]) if val > limit)
#        idx=idx+len(df2)
#        #slopes=slopes_time_volume_over_volume(flow[idx:],time[idx:])
#        volume=volume_from_flow(flow[idx:],time[idx:])
#        t1=time[idx:]-time[idx]
#        t_V=np.array(t1[1:])/np.array(volume[1:])
#        density=np.zeros(len(df2)+len(df3))
#        density[:len(df2)]=np.array(df2[analogdensity])
#        density[len(df2):]=np.array(df3[analogdensity])
#        density=np.mean(density)
#        residual_humidity=residual_humidity_over_time(flow,time,density,figure=False)
#        j= np.where(residual_humidity<target_humidity)[0]
#        if len(j)==0:
#                  print("target humidity  is not reached in ",filenames2[i][:-24])
#                  continue
#        else:                  
#            j=j[0]
#        j=j-idx
#        plt.figure()
#        plt.scatter(volume[1:],t_V,color='c',marker='.')
#        plt.scatter(volume[j],t_V[j-1],color='red')
#        plt.xlabel('volume  [m^3]')
#        plt.ylabel('time/volume [s m^-3]')
#        plt.title('time/volume over volume in the constant pressure region')
#
#        #plt.savefig('D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/risultati/t_V over V/t_V of target humidity/fig_'+str(i)+'.png')
#

#%%
#"""take all the slopes of the target humidity and    plot in  histogram"""     
#def plot_slope_histogram_of_target_humidity(filenames2,filenames3,path,target_humidity,n_points_mean_gradient=10): 
#    slope=[]
#    for i in range(len(filenames2)):
#        df2,df_p2=df_from_phase_bson(filenames2[i],path)
#        df3,df_p3=df_from_phase_bson(filenames3[i],path)
#        flow=np.zeros(len(df2)+len(df3))
#        time=np.zeros(len(df2)+len(df3))        
#        date=list(add_time_as_timeseries(df2)['timeserie'])
#        date.extend(add_time_as_timeseries(df3)['timeserie'])
#        flow[:len(df2)]=np.array(df2[analogflow])
#        flow[len(df2):]=np.array(df3[analogflow])
#        t0=time_to_num(date[0])
#        j=0
#        for t in date:
#            time[j]=time_to_num(t)-t0
#            j=j+1
#        limit=max(np.array(df3[analogpressure]))-1   #limit after that i consider constant the pressure
#        idx = next(x for x, val in enumerate(df3[analogpressure]) if val > limit)
#        idx=idx+len(df2)
#        #slopes=slopes_time_volume_over_volume(flow[idx:],time[idx:])
#        volume=volume_from_flow(flow[idx:],time[idx:])
#        t1=time[idx:]-time[idx]
#        t_V=np.array(t1[1:])/np.array(volume[1:])
#        density=np.zeros(len(df2)+len(df3))
#        density[:len(df2)]=np.array(df2[analogdensity])
#        density[len(df2):]=np.array(df3[analogdensity])
#        density=np.mean(density)
#        residual_humidity=residual_humidity_over_time(flow,time,density,figure=False)
#        j= np.where(residual_humidity<target_humidity)[0]
#        if len(j)==0:
#                  print("target humidity  is not reached in ",filenames2[i][:-24])
#                  continue
#        else:                  
#            j=j[0]
#        j=j-idx
#        slopes=np.gradient(t_V,volume[1:])
#        if n_points_mean_gradient==0 or n_points_mean_gradient==1:
#            slope.append(slopes[j])
#        else:
#            n_points_mean_gradient=int(n_points_mean_gradient/2)
#            mean_slope=np.mean(slopes[j-n_points_mean_gradient:j+n_points_mean_gradient])
#            slope.append( mean_slope)
#    plt.hist(slope)
#    plt.xlabel('Slope')
#    plt.title('Slope of t-V over V curve with target humidity of '+str(target_humidity))
#    return slope
#%%
#def slope_ratio_of_target_humidity(df2,df3,target_humidity,starting_points=200,n_points_mean_gradient=0):
#    flow=np.zeros(len(df2)+len(df3))
#    time=np.zeros(len(df2)+len(df3))        
#    date=list(add_time_as_timeseries(df2)['timeserie'])
#    date.extend(add_time_as_timeseries(df3)['timeserie'])
#    flow[:len(df2)]=np.array(df2[analogflow])
#    flow[len(df2):]=np.array(df3[analogflow])
#    t0=time_to_num(date[0])
#    j=0
#    for t in date:
#        time[j]=time_to_num(t)-t0
#        j=j+1
#    limit=max(np.array(df3[analogpressure]))-1
#    idx = next(x for x, val in enumerate(df3[analogpressure]) if val > limit)
#    idx=idx+len(df2)
#    volume=volume_from_flow(flow[idx:],time[idx:])
#    t1=time[idx:]-time[idx]
#    t_V=np.array(t1[1:])/np.array(volume[1:])
#    slopes=np.gradient(t_V,volume[1:])
#    starting_slope=sp.stats.linregress(volume[:starting_points],t_V[:starting_points])[0]
#    density=np.zeros(len(df2)+len(df3))
#    density[:len(df2)]=np.array(df2[analogdensity])
#    density[len(df2):]=np.array(df3[analogdensity])
#    density=np.mean(density)
#    residual_humidity=residual_humidity_over_time(flow,time,density,figure=False)
#    j= np.where(residual_humidity<target_humidity)[0]
#    if len(j)==0:
#              #print("target humidity  is not reached in ",phase2[i][:-24])
#              return 0
#    else:                  
#        j=j[0]
#    j=j-idx
#    if n_points_mean_gradient==0 or n_points_mean_gradient==1:
#        return slopes[j]/starting_slope
#    else:
#        n_points_mean_gradient=int(n_points_mean_gradient/2)
#        mean_slope=np.mean(slopes[j-n_points_mean_gradient:j+n_points_mean_gradient])
#        return mean_slope/starting_slope
##short test pipeline 



    


#%%
#def density_over_time(files2,files3,path,figure=False):
#    densities=[]
#    times=[]
#    for i in range(len(files3)):
#        df2,df_p2=df_from_phase_bson(files2[i],path)
#        df3,df_p3=df_from_phase_bson(files3[i],path)
#        d=np.zeros(len(df2)+len(df3))
#        t=np.zeros(len(df2)+len(df3))
#        d[:len(df2)]=np.array(df2[analogdensity])
#        d[len(df2):]=np.array(df3[analogdensity])
#        t[:len(df2)]=add_time_as_number(df2,'timestamp')['Time number']
#        t[len(df2):]=add_time_as_number(df3,'timestamp')['Time number']+t[len(df2)-1]
#        densities.append(d)
#        times.append(t)
#        if figure:
#            plt.figure()
#            plt.scatter(t,d)
#            plt.ylim([0,max(d)+0.02])
#    return densities,times
#
#"""END RESIDUAL HUMIDITY ANALYSIS"""







   
#%%
#
#"""PRELIMINARY AND GENEIRC FUNCTIONS AND SEPARATION IN CYCLES"""
#
#
#
#
#def volume_density_bson(files,path):
#    volume=[]
#    density=[]
#    for file in files:
#        df,df_phase=df_from_phase_bson(file, path)
#        volume.append(max(np.array(df_phase['PhaseVars.phaseVariable6'])))
#        density.append(max(np.array(df_phase['PhaseVars.phaseVariable9'])))
#    plt.scatter(volume,density)
#    return(volume,density)
#def time_density_feeding_bson(files,path):
#    time=[]
#    density=[]
#    finalfeeding=[]
#    initialfeeding=[]
#    for file in files:
#        df,df_phase=df_from_phase_bson(file, path)
#        time.append(max(np.array(df_phase['PhaseVars.phaseVariable1'])))
#        density.append(max(np.array(df_phase['PhaseVars.phaseVariable9'])))
#        finalfeeding.append(max(np.array(df_phase['PhaseVars.phaseVariable3'])))
#        initialfeeding.append(max(np.array(df_phase['PhaseVars.phaseVariable2'])))
#    return(time,initialfeeding,finalfeeding,density)
#
#def selected_time_density(time,density,initialfeeding,finalfeeding,limits_initialfeedig=[200,230],limits_final=[3,10]):
#    d=[]
#    t=[]
#    in_fed=[]
#    fin_fed=[]
#    for i in range(len(time)):
#        if(initialfeeding[i]>=limits_initialfeedig[0] and initialfeeding[i]<=limits_initialfeedig[1] 
#        and finalfeeding[i]>=limits_final[0] and finalfeeding[i]<=limits_final[1]):
#            t.append(time[i])
#            d.append(density[i])
#            in_fed.append(initialfeeding[i])
#            fin_fed.append(finalfeeding[i])
#    x=np.zeros([len(d),4])
#    x[:,0]=t
#    x[:,1]=in_fed
#    x[:,2]=fin_fed
#    x[:,3]=d
#    return x
#"""calculate the daily mean of a cartain variable"""
#def mean_by_day(arr,dates):
#    day=dates[0].day
#    means=[]
#    meanday=[]
#    st_err=[]
#    dates2=[]
#    for i in range(len(dates)):
#        if dates[i].day==day:
#            meanday.append(arr[i])
#        if dates[i].day!=day:
#            means.append(np.mean(meanday))
#            st_err.append(np.std(meanday))
#            meanday=[]
#            meanday.append(arr[i])
#            dates2.append(datetime.datetime(dates[i-1].year,dates[i-1].month,dates[i-1].day))
#            day=dates[i].day
#        if i==(len(dates)-1):
#            means.append(np.mean(meanday))
#            st_err.append(np.std(meanday))
#            dates2.append(datetime.datetime(dates[i].year,dates[i].month,dates[i].day))
#    return means,st_err,dates2
#            

#%%
#"""analysis of feeding and pressure in a single phase""" 
#
#def feeding_volume_in_a_phase(phase_file,path):
#    df,df_phase=df_from_phase_bson(phase_file,path)
#    df=add_time_as_number(df[['timestamp','Analogs.analog1']],'timestamp')
#    volume=volume_from_flow(df['Analogs.analog1'],df['Time number'])
#    plt.figure()
#    plt.scatter(volume[20:-5],1/df['Analogs.analog1'][20:-5])
#    
#    
#
##%%

#mappo le due variabili, se una è il tempo solo variabile tempo, altrimenti faccio x-y, x-t e y-t
#def mapping_var_big_than(df,namevar,namevar2,limit=None,timename='_time',title='correlation time-variables'):
#   if limit==None:
#       df1=df 
#   else:
#       df1=df.loc[df[namevar]>limit]
#   y=df1[namevar]
#   if (namevar2=='_time'):
#       x= numbers_from_time(df1,namevar2)
#       plt.figure()
#       plt.scatter(x, df1[namevar],marker='.')
#   else:
#       time=numbers_from_time(df1,timename)
#       x=df1[namevar2]
#       fig, axs = plt.subplots(2, 2)
#       fig.suptitle(title)
#       axs[0,0].scatter(x, y,marker='.')
#       axs[0,0].set_title(namevar+' - '+namevar2)
##       plt.xlabel(namevar2)
##       plt.ylabel(namevar)
##       plt.figure()
#       axs[0,1].scatter(time, y,marker='.')
#       axs[0,1].set_title(namevar+' - time')
##       plt.xlabel('time')
##       plt.ylabel(namevar)
##       plt.figure()
#       axs[1,0].scatter(time, x,marker='.')
#       axs[1,0].set_title(namevar2+' - time')
##       plt.xlabel('time')
##       plt.ylabel(namevar2)
###       
#
#
##%%

#calc volume as itegral of flow

def volume_from_flow(flows,times):
    """
    This function calculate the pumped volume as integral of the flow in the time.
    
    Parameters:
        flows:array of double containing the instant flows, as m^3/h
        times:array of double containing the times (in seconds) at which each flow is measured. it must be that len(flows)==len(times)
    Return:
        list of double, the volumes calculated as integral of flow in time. The first value is 0 
    """
    volume=[]
    volume.append(0)
    for i in range(0,len(flows)-1):
        volume.append(volume[i]+(flows[i+1]+flows[i])*(times[i+1]-times[i])/(2*3600))
    return volume

#%%
##voglio selezionare una zona di punti, idealmente quelli di un ciclo, che sanno sopra un certo valore eludendo le fluttuazioni"""
#def selecting_cycle(arr, limit, n_point_under,n_point_over):
#    #arr=array of values
#    #limit=lower limit in which points are not good
#    #n_point_under=number of point under which no start cycle index is taken
#    #n_point_over=number of point under which no end cycle index is taken
#    #return=starting and ending cycle indices 
#        if(n_point_under<1):
#            raise TypeError("n_point_under should be at least 1")
#        if(n_point_over<1):
#            raise TypeError("n_point_over should be at least 1")
#        if(n_point_under>len(arr)):
#            raise TypeError("n_point_under could not be more than the length of arr")
#        if(n_point_over>len(arr)):
#            raise TypeError("n_point_over could not be more than the length of arr")
#        if len(arr)<1:
#            raise TypeError("arr is empty")
#        indices=[]
#        index_start=0
#        index_end=0
#        
#
#        k=False #k is a variable of status
#        count1=0 #count1 count the numbers over limit
#        count2=0 #count1 count the numbers under limit
#        for i in range(0,len(arr)):
#            if arr[i]>limit:
#
#                count1=count1+1
#                count2=0
#            else:
#
#                count1=0
#                count2=count2+1
#            if (count1==n_point_under  and not(k)):
#                index_start=i-n_point_under+1
#                k=True
#            if (count2==n_point_over and k):
#                index_end=i-n_point_over
#                indices.append([index_start,index_end])
#                index_start=0
#                index_end=0
#                k=False
#        return indices
#
#"""con  n_point_under=3, e  n_point_over=5 funziona abbastanza bene"""
#
##%%
#
#def plot_2_var_interpoled(var1,var2,time1,time2):
#    var2_interp=np.interp(time1,time2,var2)
#    plt.figure()
#    plt.scatter(time1,var2_interp,marker='.')
#    plt.scatter(time2,var2,marker='o',alpha=0.2)
#    plt.figure()
#    plt.scatter(var2_interp,var1)
#
##%%
#"""scattr plot of all division of the df"""
#def scatter_plot_cycles(df,indices,y_var,x_var='Time number'):
#    for index in indices:
#        plt.figure()
#        plt.scatter(df[x_var][index[0]:index[1]],df[y_var][index[0]:index[1]],marker='.')
#        
#        
##%%
#"""analisi densità-volume"""
#"""density,time and flow should be array"""
#def density_volume(density,flows,time1,time2):
#    index_flow=selecting_cycle(flows,1.,3,5) #divide the points of flow in cycles in which flow is not null avoiding fluctuations
#    density_interp=np.interp(time1,time2,density) #linear interpolation of density (less sample) with the time of flow (more sample)
#    """time1 is the time whit higher number of point, time 2 less point"""
#    volumes=[]
#    densities=[]
#    for index in index_flow:
#        volume=volume_from_flow(flows[index[0]:index[1]],time1[index[0]:index[1]]) #calulate volumes as integral of flow in time
#        volumes.append(volume)
#        densities.append(density_interp[index[0]:index[1]])
#        plt.figure()
#        plt.scatter(volume,density_interp[index[0]:index[1]] , marker='.')
##        plt.figure()
##        plt.scatter(time1[index[0]:index[1]],volume, marker='.')
#    return(densities, volumes)
##%%
#def integrals_of_cycles(arr,time,cycle_index):
#    total_integral=np.zeros(len(cycle_index))
#    time_integral=np.zeros(len(arr))
#    j=0
#    for index in cycle_index:
#        total_integral[j]=np.trapz(arr[index[0]:index[1]],x=time[index[0]:index[1]])
#        j=j+1
#        for i in range(index[0],index[1]):
#            if i==0:
#                time_integral[i]=0    
#            time_integral[i]=time_integral[i-1]+np.trapz([arr[i-1],arr[i]],x=[time[i-1],time[i]])
#    return total_integral,time_integral
#
##%%
#"""divide in cycle using total feeding time=analog 21"""
##
#def select_cycle2_time(df,var,timename='Time number'):
#    j=0 
#    #j= variabile di ciclo: 0= ciclo spento, 1=ciclo acceso
#    start_i=[]
#    end_i=[]
#    times_start=[]
#    times_end=[]
#    for i in range(0,len(df[var])):
#        if (df[var][i]!=0 and not(np.isnan(df[var][i])) and j==0):
#            start_i.append(i)
#            j=1
#        if ((df[var][i]==0 or np.isnan(df[var][i])) and j==1):
#            end_i.append(i)
#            j=0
#    for i in range (0,len(end_i)):
#        times_start.append(df[timename][start_i[i]])
#        times_end.append(df[timename][end_i[i]])
#    return (times_start,times_end)
#
#
#
#def select_cycles_indices_by_time(df, times_start,times_end, timename='Time number'):
#    indices=[]
#    for i in range(0,len(times_start)):
#        df1=df.loc[df[timename]<=times_end[i]]
#        df1=df1.loc[df1[timename]>=times_start[i]]
#        indices.append([df1.index[0]+1,df1.index[len(df1)-1]+1])
#    return indices
#
##%%
#
#def final_feeding_delivery_on_pressure(df,indices,limit_pressure):
#    feed_var='analogFast1'
#    pressure_var='analogFast3'
#    for i in range(0,indices[1]-indices[0]):
#        j=indices[1]-i
#        if df[pressure_var][j]>limit_pressure:
#            return df[feed_var][j-5]
#
#
##%%
#def plot_with_same_sampling(df,namevars,path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/immagini generate"):
#    for name in namevars:
#        for name2 in namevars:
#            if name!=name2:
#                plt.figure()
#                plt.scatter(df[name],df[name2],marker='.')
#                plt.xlabel(name)
#                plt.ylabel(name2)
#                plt.title(name+'-'+name2)
#                plt.savefig(path+"/"+name+"-"+name2)
#                plt.close()
#                
#                
#def plot_with_different_sampling(df1,df2,var1,var2,timename='Time number',
#                                     path="D:/lucaz/OneDrive/Desktop/tirocinio/lavoro/immagini generate"):
#    time1=df1['Time number']
#    time2=df2['Time number']
#    for name1 in var1:
#        for name2 in var2:
#            var2_interp=np.interp(time1,time2,var2)
#            fig, axs = plt.subplots(2)
#            axs[0]=plt.scatter(time1,var2_interp,marker='.')
#            axs[0]=plt.scatter(time2,var2,marker='o',alpha=0.2)
#            axs[1]=plt.scatter(var2_interp,var1)
#
#
#                
##%%
#            
#"""ENERGIES AND POWERS"""
#
#def power_diff(power_df):
#    return power_df['load1']-power_df['load2']-power_df['load3']-power_df['load4']-power_df['load5']-power_df['load6']-power_df['load7']-power_df['load8']
#
#def power_error_diff(percentual_err,power_df):
#    err=np.zeros(len(power_df))
#    for i in range(1,9):
#        err=err+np.array(power_df['load'+str(i)])*percentual_err
#    return err
#        
#    
#    
#    
#def power_errors(percentual_error,df):
#    return np.array(df)*percentual_error
#
#
#
#def energy_from_power(powers,times):
#    energy=np.zeros(len(powers))
# 
#    for i in range(1,len(powers)):
#        energy[i]=energy[i-1]+(powers[i]+powers[i-1])*((times[i]-times[i-1])/3600)*0.5
#    return energy
#
#def final_energy_error(power_error,times):
#    total_error=0
#    for i in range(len(power_error)-1):
#        total_error=total_error+(power_error[i]+power_error[i+1])*((times[i+1]-times[i])/3600)*0.5
#    return total_error
#
#
#"""this function allow to select in a dataframe with dates, the data between 2 specific dates"""
#"""it returns the section of dataFrame between the two dates""" 
#def df_select_from_time(df,start_date,end_date,time_column_name='_time'):
#    return df[(df[time_column_name]>start_date) & (df[time_column_name]<end_date)]
#    
#def energy_pie_chart(powerdf,timename='Time number',pathsave=path3, savefigure=False,name='piechart'):
#    energies=[]
#    if not(timename in powerdf.columns):
#        powerdf=add_time_as_number2(powerdf)
#    labels=['hydraulic unit', 'washing pump','compressor', 'filter feeding pump',' plates moving', 'recirculation pump'  ]
#    colors=['blue','orange','green','red','cyan','pink']
#    explode=(0.4,0.4,0.4,0,0.4,0.4)
#    for i in range(2,8):
#        load='load'+str(i)
#        energies.append(energy_from_power(np.array(powerdf[load]),np.array(powerdf['Time number']))[-1])
#
#    tot_en=sum(np.array(energies))
#    if tot_en<1:
#        nenergies=np.array(energies)*1000
#        tot_en=tot_en*1000
#        plt.figure()
#        patches=plt.pie(nenergies,explode=explode,autopct='%1.1f%%', shadow=False,colors=colors, startangle=170)
#        plt.legend(patches[0], labels, loc="lower right")
#        
#        plt.title('total energy = '+str(int(tot_en))+' [Wh]')
#        plt.tight_layout()
#    else:    
#        plt.figure()
#        patches=plt.pie(energies,explode=explode,autopct='%1.1f%%', shadow=False,colors=colors, startangle=170)
#        plt.legend(patches[0], labels, loc="lower right")
#        
#        plt.title('total energy = '+str(int(tot_en))+' [kWh]')
#        plt.tight_layout()
#    #plt.axis('equal')
#    if savefigure:
#        plt.savefig(pathsave+'/'+name+'.png')
#        plt.close()
#    return energies
#
#
#def total_time(start_date,end_date):
#    return time_to_num(end_date)-time_to_num(start_date)
#
#"""this function will furnish all the energies of each motor for each cycle"""
#"""it requires the power dataFrame and the dates of starting and ending cycles"""
#"""it will funish also the error"""
#def all_energies_for_cycles(power_df,starting_dates,ending_dates):
#    energies=[]
#    error=[]
#    if not('Time number' in power.columns):
#        power_df=add_time_as_number2(power_df)
#    for i in range(len(starting_dates)):
#        df=df_select_from_time(power_df,starting_dates[i],ending_dates[i])
#        all_energies=[]
#        all_errors=[]
#        for j in range(1,8):
#            load='load'+str(j)
#            all_energies.append(energy_from_power(np.array(df[load]),np.array(df['Time number']))[-1])
#            all_errors.append(final_energy_error(power_errors(0.01,df[load]),np.array(df['Time number'])))
#        energies.append(all_energies)
#        error.append(all_errors)
#
#    return energies,error
#
#def energies_diff(energies,energy_errors):
#    en_diff=energies[0]
#    en_err=energy_errors[0]
#    for i in range(1,len(energies)):
#        en_diff=en_diff-energies[i]
#        en_err=en_err+energy_errors[i]
#    return en_diff,en_err
###%%
#    
##"""short pipeline"""
#json_file_to_df(read_json_names(pathEventLog),pathEventLog)
#json_file_to_df(read_json_names(pathCycleLog),pathCycleLog)
#json_file_to_df(read_json_names(path2),path2)
#
#
##%%
#powers=[Power,Power_1,Power_2,Power_3,Power_4,Power_5,Power_6,Power_7,Power_8,Power_9,Power_10]
#energy=[]
#error=[]
#date=[]
#mean_powers=[]
#power=pd.DataFrame()
#for p in powers:
#    power=pd.concat([power,p])
#power=power.reset_index(drop=True)
#
#
#if not('Time number' in power.columns):
#    power=add_time_as_number2(power)
#%%
#j=70
#for i in range(j,len(CycleLog)):
#    tstart=timefromiso(CycleLog['timeFrom'][i])
#    tend=timefromiso(CycleLog['timeTo'][i])
#    time=total_time(tstart,tend)/3600
#    try:
#        df=df_select_from_time(power,tstart,tend)
#        energy.append(energy_from_power(np.array(df['load1']),np.array(df['Time number']))[-1])
#        pathsave=path+'/risultati/powers/pie chart'
#        energies=energy_pie_chart(df,pathsave=pathsave,name=str(i))
#        plt.close()
#    except:
#        j=i
#        continue
#    mean_powers.append(np.array(energies)/time)
#    power_error=power_errors(0.01,df['load1'])
#    error.append(final_energy_error(power_error,np.array(df['Time number'])))
#    date.append(tstart)
#
##%%
#starting_date=add_time_as_timeseries(CycleLog[70:],'timeFrom')['timeserie']
#ending_date=add_time_as_timeseries(CycleLog[70:],'timeTo')['timeserie']
#energies,errors=all_energies_for_cycles(power,np.array(starting_date)[:197],np.array(ending_date)[:197])
##%%
#cycles=list(range(197))
##cicli_da_escludere=[70,72,74,76,78,81,85,108,119,147,148,149,155,156,157,164,246,261,262]
#for i in range(len(cicli_da_escludere)):
#    
#    cycles.remove(cicli_da_escludere[i]-70)
#energy_of_interest=[0,1,3,4,5,6]
#labels=['line motor','hydraulic unit', 'washing pump','compressor', 'filter feeding pump',' plates moving', 'recirculation pump'  ]
#for j in energy_of_interest:
#    plt.figure()
#    for i in cycles:
#
#        plt.scatter(i,energies[i][j],marker='.')
#        plt.errorbar(i,energies[i][j],errors[i][j],fmt='.k')
#    plt.title(labels[j])
#
##%%
#energy_diff=[]
#energy_diff_error=[]
#for i in cycles:
#    en,err=energies_diff(energies[i],errors[i])
#    energy_diff.append(en)
#    energy_diff_error.append(err)
#plt.figure()
#plt.scatter(np.array(cycles)+70,energy_diff,marker='.')
#plt.errorbar(np.array(cycles)+70,energy_diff,energy_diff_error,fmt='.')
#plt.title('Difference between line energy and all other motor energy for cycle')
#plt.ylabel('Energy [kWh]')
#plt.xlabel('Cycle')
##%%
#energy_line=[]
#line_err=[]
#for i in cycles:
#    energy_line.append(energies[i][0])
#    line_err.append(errors[i][0])
#percentage_energy_lost=np.array(energy_diff)/np.array(energy_line)*100
#perc_err=(np.array(line_err)/np.array(energy_line)+np.array(energy_diff_error)/np.array(energy_diff))
#perc_err=perc_err*percentage_energy_lost
#plt.scatter(np.array(cycles)+70,percentage_energy_lost,marker='.',alpha=0.5)
#plt.errorbar(np.array(cycles)+70,percentage_energy_lost,perc_err,fmt='.')
#plt.title('Percentage energy lost')
#plt.ylabel('Percentage [%]')
#plt.xlabel('Cycles')
#
#
##%%
#percentage2=np.copy(percentage_energy_lost)
#for j in range(len(percentage_energy_lost)):
#    if percentage_energy_lost[j]>7:
#        df=df_select_from_time(power,timefromiso(CycleLog['timeFrom'][cycles[j]+70]),timefromiso(CycleLog['timeTo'][cycles[j]+70]))
#        df=df[-400:-1]
#        energy_lost=energy_from_power(np.array(power_diff(df)),np.array(df['Time number']))[-1]
#        energyline1=energy_from_power(np.array(df['load1']),np.array(df['Time number']))[-1]
#        percentage2[j]=energy_lost/energyline1*100
##%%        
#_,bins,_=plt.hist((percentage2-np.mean(percentage2))/np.std(percentage2),bins=20,alpha=.5,density=1)
#mu=np.mean(percentage2)
#sigma=np.std(percentage2)
#best_fit_line = sp.stats.norm.pdf(bins, 0, 1)
#plt.plot(bins, best_fit_line)
