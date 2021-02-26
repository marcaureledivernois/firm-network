import numpy as np
from mutualinfo import entropy, entropy_gaussian, mutual_information, directed_information,\
    directed_information_gaussian, directed_information_in_network, conditioning_correl, \
    directed_information_in_network_with_pf,  max_corr_pf,\
    conditioning_with_one_pf, conditioning_with_trueparents, directed_information_gaussian_with_pf, conditioning_with_rnn
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import scipy.linalg as la
import pickle
import random


#================================ LOAD DATA ============================================================================

datacsv = 'dailySP - 01Jan2010 - 31Dec2014_clean.csv'
data = pd.read_csv(datacsv)

RF = pd.read_csv('RF.csv',sep=',')

#================================ KEEP BIGGEST MARKET CAPS =============================================================


def idx_N_biggest_firms(N):
    return data.groupby(['PERMCO'], sort=False)['market_cap'].max().sort_values(ascending=False).head(N).index.to_list()


data = data[data['PERMCO'].isin(idx_N_biggest_firms(1000))]   #DIG will be computed using the data from the 1000 biggest firms

data['simple_return'] = data.groupby('PERMCO').adj_price.pct_change()
data['log_return'] = np.log(1 + data.simple_return)

data = pd.merge(data, RF, on='date', how='left')

data['excess_return'] = data['log_return'] - data['rf']


#================================ DATA TRANSFORM =======================================================================

data_T = pd.pivot_table(data, values="excess_return", index="date", columns="PERMCO")
# data_T = data_T.dropna(thresh=50, axis=1)   #remove firms that has less than 50 days of data

nan_firms = data_T.isna().any()[data_T.isna().any()==True].index.to_list()

data_T = data_T.dropna(axis=1)  # remove firms with nan in their time series


firmlist = pd.DataFrame(list(data_T.columns), columns=['PERMCO'])
firmlist = firmlist.merge(data[['COMNAM', 'PERMCO', 'TICKER']], on='PERMCO')
firmlist = firmlist.drop_duplicates(subset='PERMCO', keep='last').reset_index(drop=True)

#data_T.to_csv("data_T.csv", index=False)

def idx_N_biggest_firms_removenan(N):
    data_nonan = data.drop(data[data['PERMCO'].isin(nan_firms)].index)
    return data_nonan.groupby(['PERMCO'], sort=False)['market_cap'].max().sort_values(ascending=False).head(N).index.to_list()

#compute the DI for the 30 biggest firms (using the data of 1000 biggest firms)
firmlist['big']= firmlist['PERMCO'].isin(idx_N_biggest_firms_removenan(30))
idx = firmlist.index[firmlist['big']].to_list()

horizon = 5

#===================================== BUILD VARIABLES MATRIX  =========================================================

variables = data_T.iloc[:horizon, :].T.stack(dropna=False).reset_index(name='new')['new'].T
for i in range(1, int(data_T.shape[0] / horizon)):
    variables = pd.concat([variables, data_T.iloc[i * horizon: i * horizon + horizon, :].T.stack(
        dropna=False).reset_index(name='new')['new']], axis=1)
variables = variables.T.reset_index(drop=True)
variables = np.asarray(variables)

#========================================== DIG (KNN ESTIMATOR) ========================================================

#condition on RNN
t0 = time.time()
DIG_sim_rnn = np.asarray([[directed_information_in_network_with_pf(variables, horizon, i, j, conditioning_with_rnn(
    data_T,i,j,firmlist,horizon)['x_set_k']) for j in idx] for i in idx])
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes') #

#============================================== save ===================================================================

filesave = 'DIG_empirical_' + datacsv[-14:-10] + '.pkl'

with open(filesave, 'wb' ) as f:
    pickle.dump([DIG_sim_rnn, firmlist], f)

