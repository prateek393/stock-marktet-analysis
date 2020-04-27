# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 13:17:47 2019

@author: Ankit
"""
import datetime as dt
import matplotlib.pyplot as plt 
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import pickle
import os

import urllib.request
from bs4 import BeautifulSoup
import requests
import numpy as np
import re

style.use('ggplot')

def heat_map():
    df = pd.read_csv('joined_adj_close.csv')
    df_correlation = df.corr()
    
    data = df_correlation.values
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    heatmap = ax.pcolor(data,cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    #set the points where the labels will finally be put
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.invert_yaxis() # removes gap above
    ax.xaxis.tick_top()
    
    coloumn_labels=df_correlation.columns
    row_labels = df_correlation.index
    #set what is to be written on x and y axis
    ax.set_xticklabels(coloumn_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation= 90)    #rotation of the lable names on axis
    heatmap.set_clim(-1,1)   #legend for heatmap
    
    plt.tight_layout()
    plt.show
    
    
    
heat_map()