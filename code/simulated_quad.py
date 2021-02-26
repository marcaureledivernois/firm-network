import numpy as np
from mutualinfo import entropy, entropy_gaussian, mutual_information, directed_information,\
    directed_information_gaussian, directed_information_in_network, conditioning_correl, \
    directed_information_in_network_with_pf,  max_corr_pf,\
    conditioning_with_one_pf, conditioning_with_trueparents, conditioning_with_rnn, conditioning_with_A, \
    conditioning_with_multiple_pf, directed_information_gaussian_with_pf, directed_information_in_network_corr_and_pf
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import scipy.linalg as la
import pickle
import random
from sklearn.preprocessing import StandardScaler

nb_assets = 15
nb_parents = 10
nb_periods = 10000
A = np.zeros([nb_assets,nb_assets,nb_assets]) #elements A[x,y,i]+A[y,x,i] represent interaction effect of x and y in the process i
epsilon = 0.6

A_tp = np.zeros([nb_assets,nb_assets])

for i in range(A.shape[2]):
    start = random.sample(range(0, nb_assets), nb_parents)
    A[start, : ,i] = np.random.uniform(low=-0.9, high=0.9, size=(A[start, : ,i].shape))  # generate influence
    A[:, start, i] = np.random.uniform(low=-0.9, high=0.9, size=(A[:, start, i].shape))  # generate influence
    lowinf = np.abs(np.sum((A[:,:,i] + A[:,:,i].T) ,axis=1)) < 1*np.mean(np.abs(np.sum((A[:,:,i] + A[:,:,i].T) ,axis=1)))
    A[lowinf,:,i] = 0
    A[:, lowinf, i] = 0
    #A[:,:,i] = A[:,:,i] / (max(abs(la.eig(A[:,:,i])[0])) + 0.1)    #stationarity of A
    A_tp[i,:] = 1*(np.sum((A[:,:,i] + A[:,:,i].T) ,axis=1) != 0)    #if A[i,j] = 1, it means j influences i

b = np.ones([nb_assets]) * 0.7

X = np.ones([nb_periods,nb_assets])
for t in range(1,nb_periods):
    for i in range(nb_assets):
            pX = np.power(np.abs(X[t-1,:]),1/3)*np.sign(X[t-1,:])
            X[t,i] = b[i] * np.dot(pX, np.dot(A[:,:,i], pX.T )) + 0.2 * np.random.randn()
            X[t,i] = np.where(X[t,i] < -50, -50, X[t,i])
            X[t, i] = np.where(X[t, i] > 50, 50, X[t,i])

X = np.delete(X, 0, 0)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X)

horizon = 5

iter = list(range(A.shape[0]))

firmlist = pd.DataFrame(data={'PERMCO': iter, 'COMNAM': iter, 'TICKER': iter})

#if A[i,j] = 1, it means j influences i
#if DIG[i,j] = 1, it means i influences j

#===================================== BUILD VARIABLES MATRIX  =========================================================

variables = X.iloc[:horizon, :].T.stack(dropna=False).reset_index(name='new')['new'].T
for i in range(1, int(X.shape[0] / horizon)):
    variables = pd.concat([variables, X.iloc[i * horizon: i * horizon + horizon, :].T.stack(
        dropna=False).reset_index(name='new')['new']], axis=1)
variables = variables.T.reset_index(drop=True)
variables = np.asarray(variables)

#============================================= DIG =====================================================================

#condition on true parents
t0 = time.time()
DIG_sim_tp = np.asarray([[directed_information_in_network(variables, horizon, i, j, conditioning_with_trueparents(A_tp, i , j))
                       for j in range(X.shape[1])] for i in range(X.shape[1])])              #data_T.shape[1]
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')   # 372.3340076208115 minutes

#condition on highest correlated nodes
t0 = time.time()
DIG_sim_corr = np.asarray([[directed_information_in_network(variables, horizon, i, j, conditioning_correl(
    X, firmlist, i, j, nb_highest_corr=13)) for j in range(X.shape[1])] for i in range(X.shape[1])])              #data_T.shape[1]
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')  # 345.8092816670736 minutes

#condition on first portfolio
t0 = time.time()
DIG_sim_pf = np.asarray([[directed_information_in_network_with_pf(variables, horizon, i, j, conditioning_with_one_pf(
    variables, X, A, i, j, firmlist , horizon)['x_set_k']) for j in range(X.shape[1])] for i in range(X.shape[1])])
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')  # 15.34332121213278 minutes

#condition on RNN
t0 = time.time()
DIG_sim_rnn = np.asarray([[directed_information_in_network_with_pf(variables, horizon, i, j, conditioning_with_rnn(
    X,i,j,firmlist,horizon)['x_set_k']) for j in range(X.shape[1])] for i in range(X.shape[1])])
np.fill_diagonal(DIG_sim_rnn,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes') #

#condition on 9 highest corr and 1 projection
t0 = time.time()
DIG_sim_corr_and_pf = np.asarray([[directed_information_in_network_corr_and_pf(variables, horizon, i, j, conditioning_correl(
    X, firmlist, i, j, nb_highest_corr=9) , conditioning_with_one_pf(variables, X, A, i, j, firmlist , horizon)['x_set_k'])
                          for j in range(X.shape[1])] for i in range(X.shape[1])])
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')  # 15.34332121213278 minutes

#================================================= Save  ===============================================================

#with open( 'DIG_sim_quad4.pkl', 'wb' ) as f:
#    pickle.dump([DIG_sim_tp,DIG_sim_corr,DIG_sim_pf,DIG_sim_rnn, X, A, A_tp], f)

#================================================= load  ===============================================================

DIGS = pickle.load(open( 'DIG_sim_quad3.pkl', 'rb' ))
DIG_sim_tp = DIGS[0]
DIG_sim_corr = DIGS[1]
DIG_sim_pf = DIGS[2]
DIG_sim_rnn = DIGS[3]
X = DIGS[4]
A = DIGS[5]
A_tp = DIGS[6]

#============================================= DIG_binary ==============================================================

def DIG_binary(DIG, threshold, axis=None):
    if axis==None:
        DIG_b = (DIG > threshold)*1
    else:
        DIG_b = (DIG > threshold) * 1
    return DIG_b

DIG_binary_tp = DIG_binary(DIG=DIG_sim_tp, threshold = 0, axis = None)
DIG_binary_corr = DIG_binary(DIG=DIG_sim_corr, threshold = 0, axis = None)
DIG_binary_pf = DIG_binary(DIG=DIG_sim_pf, threshold =0, axis = None)
#DIG_binary_corr_and_pf = DIG_binary(DIG=DIG_sim_corr_and_pf, threshold =0, axis = None)
DIG_binary_rnn = DIG_binary(DIG=DIG_sim_rnn, threshold =0, axis = None)

#============================================= analyze DIG =============================================================

def perf_measures(DIG_b,A):
    measures = dict()
    measures['TP'] = 1*(DIG_b * (A.T != 0))
    measures['FP'] = 1*(DIG_b * (A.T == 0))
    measures['FN'] = 1*((1-DIG_b) *  (A.T != 0))
    measures['TN'] = 1*((1-DIG_b) * (A.T == 0) )
    measures['precision'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FP'].sum().sum())
    measures['recall'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FN'].sum().sum())
    measures['specificity'] = measures['TN'].sum().sum() / (measures['TN'].sum().sum() + measures['FP'].sum().sum())
    print('precision : ' , measures['precision'] , ', recall : ' , measures['recall'], ' , specificity : ' , measures['specificity'])
    return measures


perf_measures(DIG_binary_tp,A_tp)
perf_measures(DIG_binary_corr,A_tp)
perf_measures(DIG_binary_pf,A_tp)
#perf_measures(DIG_binary_corr_and_pf,A_tp)
perf_measures(DIG_binary_rnn,A_tp)

#============================================= plot =============================================================

def plot_perf(DIG, model_name, axis_mean = None):
    thresholds = np.linspace(0,DIG.max(),51)
    precisions = []
    recalls = []
    specificities = []

    for thr in thresholds:
        DIG_bin = DIG_binary(DIG=DIG, threshold = thr, axis = axis_mean)
        precisions.append(perf_measures(DIG_bin,A_tp)['precision'])
        recalls.append(perf_measures(DIG_bin,A_tp)['recall'])
        specificities.append(perf_measures(DIG_bin,A_tp)['specificity'])

    plt.style.use('seaborn-whitegrid')
    plt.plot(thresholds, precisions, '-', color='red')
    plt.plot(thresholds, recalls, '-', color='blue')
    #plt.plot(thresholds, specificities, '-', color='green')

    plt.title('Performance measures - ' + model_name)
    plt.xlabel('Threshold')
    plt.ylabel('Performance measures')
    plt.legend(('Precision','Recall','Specificity'), loc='center right')
    #plt.savefig('Linear_Performance_' + model_name + '.png')
    plt.show()


plot_perf(DIG_sim_tp, 'True parents', axis_mean = None)
plot_perf(DIG_sim_corr, 'Most correlated assets', axis_mean = None)
plot_perf(DIG_sim_pf, 'Correlated Portfolio', axis_mean = None)
plot_perf(DIG_sim_rnn, 'RNN', axis_mean = None)
#plot_perf(DIG_sim_corr_and_pf, 'Corr and Proj', axis_mean = None)


#====================== precision recall curve with sklearn ============================================================
from sklearn.metrics import precision_recall_curve

precision_tp, recall_tp, thresholds_tp = precision_recall_curve(np.reshape(1*(A_tp.T!=0),-1), np.reshape(DIG_sim_tp,-1))
precision_corr, recall_corr, thresholds_corr = precision_recall_curve(np.reshape(1*(A_tp.T!=0),-1), np.reshape(DIG_sim_corr,-1))
precision_pf, recall_pf, thresholds_pf = precision_recall_curve(np.reshape(1*(A_tp.T!=0),-1), np.reshape(DIG_sim_pf,-1))
#precision_corr_and_pf, recall_corr_and_pf, thresholds_corr_and_pf = precision_recall_curve(np.reshape(1*(A_tp.T!=0),-1), np.reshape(DIG_sim_corr_and_pf,-1))
precision_RNN, recall_RNN, thresholds_RNN = precision_recall_curve(np.reshape(1*(A_tp.T!=0),-1), np.reshape(DIG_sim_rnn,-1))


plt.plot(recall_tp, precision_tp, '-', color='blue')
plt.plot(recall_corr, precision_corr, '-', color='green')
plt.plot(recall_pf, precision_pf, '-', color='orange')
plt.plot(recall_RNN, precision_RNN, '-', color='red')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(('True Parents', 'Correlated Assets' , 'Ideal Portfolio', 'RNN'), loc='lower left')
plt.grid(True)
#plt.savefig('pre_rec_curves_gaussian.png')
plt.show()


#====================== test RNN stability ============================================================

i,j = 4,0   # i influences j
k = 5

DI_infl = []
for it in range(k):
    DI_infl.append(directed_information_in_network_with_pf(
        variables, horizon, i, j, conditioning_with_rnn( X,i,j,firmlist,horizon)['x_set_k']))

print(DI_infl)

i,j = 0,4   # i doesnt influence j

DI_noinfl = []
for it in range(k):
    DI_noinfl.append(directed_information_in_network_with_pf(
        variables, horizon, i, j, conditioning_with_rnn( X,i,j,firmlist,horizon)['x_set_k']))

print(DI_noinfl)

plt.plot(list(range(k)), DI_noinfl, color='red')
plt.plot(list(range(k)), DI_infl, color='green')
plt.show()