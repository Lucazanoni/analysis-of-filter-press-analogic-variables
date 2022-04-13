# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:40:36 2022

@author: lucaz
"""
import tirocinio
import pandas as pd
import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
testfiledir=currentdir+'/test_files'


def test_df_from_bson():
    """
    TEST:
    if function returns a pandas DataFrame
    """
    
    assert type(tirocinio.df_from_bson('bson.bson',testfiledir))==pd.DataFrame
    

def test_df_from_phase_bson():
    """
    Test:
    if function return 2 elements
    if the 2 elements are pandas DataFrame
    
    """
    assert len(tirocinio.df_from_phase_bson('phasebson.bson',testfiledir))==2
    assert type(tirocinio.df_from_phase_bson('phasebson.bson',testfiledir)[0])==pd.DataFrame
    assert type(tirocinio.df_from_phase_bson('phasebson.bson',testfiledir)[1])==pd.DataFrame
    
def test_cycle_list_file():
    """
    Test: 
    if function return a list
    """
    assert type(tirocinio.cycle_list_file(62,testfiledir))==list
    
def test_take_datetime():
    """
    Test:
    if function return a Dataframe
    if there is a column called 'time'
    
    """
    df=(tirocinio.df_from_bson('bson.bson',testfiledir))
    assert type(tirocinio.take_datetime(df,'_time'))==pd.DataFrame
    assert tirocinio.take_datetime(df,'_time')['time'] in take_datetime(df,'_time')==True
    
    
def test_time_to_num():
    

def test_