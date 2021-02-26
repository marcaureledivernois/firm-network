import numpy as np
from mutualinfo import entropy, entropy_gaussian, mutual_information, directed_information,\
    directed_information_gaussian, directed_information_in_network, conditioning_correl, \
    directed_information_in_network_with_pf,  max_corr_pf,\
    conditioning_with_one_pf, conditioning_with_trueparents, conditioning_with_rnn, conditioning_with_A, \
    conditioning_with_multiple_pf, directed_information_gaussian_with_pf
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import scipy.linalg as la
import pickle
from numpy import linalg as la


nb_parents = 10
nb_assets =  15
nb_periods = 30000

A = np.zeros([nb_assets,nb_assets]) #element i,j of A represents influence of j on i
A[:nb_parents,:nb_parents] = np.random.uniform(low=-0.9, high=0.9, size=A[:nb_parents,:nb_parents].shape)
A[nb_parents:,:] = np.random.uniform(low=-0.9, high=0.9, size=A[nb_parents:,:].shape)

small_eps = 0.4
big_eps = 0.7

A[:nb_parents,:nb_parents] = np.where(
    (A[:nb_parents,:nb_parents] < small_eps) & (A[:nb_parents,:nb_parents] > -small_eps) , 0, A[:nb_parents,:nb_parents])

A[nb_parents:,:] = np.where(
    (A[nb_parents:,:] < big_eps) & (A[nb_parents:,:] > - big_eps) , 0, A[nb_parents:,:])

np.fill_diagonal(A,0)

spectralradius = max(abs(la.eig(A)[0])) #used for stationarity of A
A = A / (spectralradius + 0.1)

x0 = np.random.randn(A.shape[0], 1)
X = np.zeros([1,A.shape[0]])


for i in range(1,nb_periods):
    eps =  np.random.randn(A.shape[0], 1)
    new_data = np.dot(A, X[i-1].T ) + eps.T
    X = np.concatenate((X, new_data), axis=0)
X = np.delete(X, 0, 0)

X = pd.DataFrame(X)

horizon = 5

iter = list(range(A.shape[0]))

firmlist = pd.DataFrame(data={'PERMCO': iter, 'COMNAM': iter, 'TICKER': iter})


#===================================== BUILD VARIABLES MATRIX  =========================================================

variables = X.iloc[:horizon, :].T.stack(dropna=False).reset_index(name='new')['new'].T
for i in range(1, int(X.shape[0] / horizon)):
    variables = pd.concat([variables, X.iloc[i * horizon: i * horizon + horizon, :].T.stack(
        dropna=False).reset_index(name='new')['new']], axis=1)
variables = variables.T.reset_index(drop=True)
variables = np.asarray(variables)


#======================================= test of computational time ====================================================

i = 0
j = 1

#condition on true parents
t0 = time.time()
print(directed_information_in_network(variables, horizon, i, j, conditioning_with_trueparents(A, i , j )))
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')

t0 = time.time()
print(directed_information_gaussian(variables, horizon, i, j, conditioning_with_trueparents(A, i , j )))
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')

#condition on highest correlated nodes
t0 = time.time()
set_k = conditioning_correl(X, firmlist,i, j, nb_highest_corr= 10)
print(directed_information_in_network(variables, horizon, i, j, set_k))
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')

t0 = time.time()
set_k = conditioning_correl(X, firmlist,i, j, nb_highest_corr= 10)
print(directed_information_gaussian(variables, horizon, i, j, set_k))
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')

#condition on first portfolio
t0 = time.time()
print(directed_information_in_network_with_pf(variables, horizon, i, j,  conditioning_with_one_pf(variables,
        X, A, i, j, firmlist , horizon)['x_set_k']))
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')

t0 = time.time()
print(directed_information_gaussian_with_pf(variables, horizon, i, j,  conditioning_with_one_pf(variables,
        X, A, i, j, firmlist , horizon)['x_set_k']))
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')

#condition on multiple pf
t0 = time.time()
print(directed_information_in_network_with_pf(variables, horizon, i, j, conditioning_with_multiple_pf(
    variables, X, A, i, j, firmlist , horizon, nb_add_pf=10)['x_set_k']))
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes') # 121 minutes

#condition on rnn
t0 = time.time()
print(directed_information_in_network_with_pf(variables, horizon, i, j, conditioning_with_rnn(
    X,i,j,firmlist,horizon)['x_set_k']))
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes') # 121 minutes


#========================================== DIG (GAUSSIAN ESTIMATOR) ===================================================

#condition on true parents
t0 = time.time()
DIG_sim_tp = np.asarray([[directed_information_gaussian(variables, horizon, i, j,
    conditioning_with_trueparents(A, i , j )) for j in range(X.shape[1])] for i in range(X.shape[1])])
np.fill_diagonal(DIG_sim_tp,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')   #

#condition on first portfolio
t0 = time.time()
DIG_sim_pf = np.asarray([[directed_information_gaussian_with_pf(variables, horizon, i, j,
    conditioning_with_one_pf(variables, X, A, i, j, firmlist , horizon)['x_set_k']) for j in range(X.shape[1])]
                         for i in range(X.shape[1])])
np.fill_diagonal(DIG_sim_pf,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')  #

#condition on highest correlated nodes
t0 = time.time()
DIG_sim_corr = np.asarray([[directed_information_gaussian(variables, horizon, i, j, conditioning_correl(
    X, firmlist, i, j, nb_highest_corr=5)) for j in range(X.shape[1])] for i in range(X.shape[1])])              #data_T.shape[1]
np.fill_diagonal(DIG_sim_corr,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')  #

#condition on RNN
t0 = time.time()
DIG_sim_rnn = np.asarray([[directed_information_gaussian_with_pf(variables, horizon, i, j, conditioning_with_rnn(
    X,i,j,firmlist,horizon)['x_set_k']) for j in range(X.shape[1])] for i in range(X.shape[1])])
np.fill_diagonal(DIG_sim_rnn,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes') #

#condition on portfolio with A weights
t0 = time.time()
DIG_sim_pf_A = np.asarray([[directed_information_gaussian_with_pf(variables, horizon, i, j, conditioning_with_A(
    X, A, i, j, firmlist , horizon)['x_set_k']) for j in range(X.shape[1])] for i in range(X.shape[1])])
np.fill_diagonal(DIG_sim_pf_A,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')  #

#condition on multiple pf
t0 = time.time()
DIG_sim_mult = np.asarray([[directed_information_gaussian_with_pf(variables, horizon, i, j, conditioning_with_multiple_pf(
    variables, X, A, i, j, firmlist , horizon, nb_add_pf=10)['x_set_k']) for j in range(X.shape[1])] for i in range(X.shape[1])])
np.fill_diagonal(DIG_sim_mult,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes') #


#========================================== DIG (KNN ESTIMATOR) ===================================================

#condition on true parents
t0 = time.time()
DIG_sim_tp = np.asarray([[directed_information_in_network(variables, horizon, i, j, conditioning_with_trueparents(A, i , j ))
                       for j in range(X.shape[1])] for i in range(X.shape[1])])              #data_T.shape[1]
np.fill_diagonal(DIG_sim_tp,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')   #

#condition on first portfolio
t0 = time.time()
DIG_sim_pf = np.asarray([[directed_information_in_network_with_pf(variables, horizon, i, j, conditioning_with_one_pf(variables,
    X, A, i, j, firmlist , horizon)['x_set_k']) for j in range(X.shape[1])] for i in range(X.shape[1])])
np.fill_diagonal(DIG_sim_pf,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')  #

#condition on portfolio with A weights
t0 = time.time()
DIG_sim_pf_A = np.asarray([[directed_information_in_network_with_pf(variables, horizon, i, j, conditioning_with_A(
    X, A, i, j, firmlist , horizon)['x_set_k']) for j in range(X.shape[1])] for i in range(X.shape[1])])
np.fill_diagonal(DIG_sim_pf_A,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')  #

#condition on highest correlated nodes
t0 = time.time()
DIG_sim_corr = np.asarray([[directed_information_in_network(variables, horizon, i, j, conditioning_correl(
    X, firmlist, i, j, nb_highest_corr=10)) for j in range(X.shape[1])] for i in range(X.shape[1])])              #data_T.shape[1]
np.fill_diagonal(DIG_sim_corr,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes')  #

#condition on multiple pf
t0 = time.time()
DIG_sim_mult = np.asarray([[directed_information_in_network_with_pf(variables, horizon, i, j, conditioning_with_multiple_pf(
    variables, X, A, i, j, firmlist , horizon, nb_add_pf=10)['x_set_k']) for j in range(X.shape[1])] for i in range(X.shape[1])])
np.fill_diagonal(DIG_sim_mult,0)
t1 = time.time()
print('time elapsed : ', (t1-t0)/60 , 'minutes') #


#============================================= load ====================================================================

DIGS = pickle.load(open( 'DIG_linear_knn.pkl', 'rb' ))
DIG_sim_tp = DIGS[0]
DIG_sim_corr = DIGS[1]
DIG_sim_pf = DIGS[2]
DIG_sim_pf_A = DIGS[3]
DIG_sim_mult = DIGS[4]
X = DIGS[5]
A = DIGS[6]

DIGS = pickle.load(open( 'DIG_linear_gaussian.pkl', 'rb' ))
DIG_sim_tp = DIGS[0]
DIG_sim_corr = DIGS[1]
DIG_sim_pf = DIGS[2]
DIG_sim_pf_A = DIGS[3]
X = DIGS[4]
A = DIGS[5]

#============================================= DIG binary ==============================================================

def DIG_binary(DIG, threshold, axis=None):
    if axis==None:
        DIG_b = (DIG > threshold)*1
    else:
        DIG_b = (DIG > threshold) * 1
    return DIG_b

DIG_binary_tp = DIG_binary(DIG=DIG_sim_tp, threshold = np.mean(DIG_sim_tp), axis = None)
DIG_binary_corr = DIG_binary(DIG=DIG_sim_corr, threshold = np.mean(DIG_sim_corr), axis = None)
DIG_binary_pf = DIG_binary(DIG=DIG_sim_pf, threshold = np.mean(DIG_sim_pf), axis = None)
DIG_binary_pf_A = DIG_binary(DIG=DIG_sim_pf_A, threshold = np.mean(DIG_sim_pf_A), axis = None)
#DIG_binary_rnn = DIG_binary(DIG=DIG_sim_rnn, threshold = 0.01, axis = None)

#============================================= analyze DIG =============================================================

def perf_measures(DIG_b,A):
    measures = dict()
    measures['TP'] = 1*(DIG_b * (A.T != 0))
    measures['FP'] = 1*(DIG_b * (A.T == 0))
    measures['FN'] = 1*((1-DIG_b) *  (A.T != 0))
    measures['TN'] = 1*((1-DIG_b) * (A.T == 0) )
    measures['precision'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() + measures['FP'].sum().sum())
    measures['recall'] = measures['TP'].sum().sum() / (measures['TP'].sum().sum() +measures['FN'].sum().sum())
    measures['specificity'] = measures['TN'].sum().sum() / (measures['TN'].sum().sum() + measures['FP'].sum().sum())
    print('precision : ' , measures['precision'] , ', recall : ' , measures['recall'], ' , specificity : ' , measures['specificity'])
    return measures

perf_measures(DIG_binary_tp,A)
perf_measures(DIG_binary_corr,A)
perf_measures(DIG_binary_pf,A)
perf_measures(DIG_binary_pf_A,A)
#perf_measures(DIG_binary_rnn,A)

#============================================= plot =============================================================

def plot_perf(DIG, model_name, axis_mean = None):
    thresholds = np.linspace(0,DIG.max(),51)
    precisions = []
    recalls = []
    specificities = []

    for thr in thresholds:
        DIG_bin = DIG_binary(DIG=DIG, threshold = thr, axis = axis_mean)
        precisions.append(perf_measures(DIG_bin,A)['precision'])
        recalls.append(perf_measures(DIG_bin,A)['recall'])
        specificities.append(perf_measures(DIG_bin,A)['specificity'])

    plt.style.use('seaborn-whitegrid')
    plt.plot(thresholds, precisions, '-', color='red')
    plt.plot(thresholds, recalls, '-', color='blue')
    #plt.plot(thresholds, specificities, '-', color='green')

    plt.title('Performance measures - ' + model_name)
    plt.xlabel('Threshold')
    plt.ylabel('Performance measures')
    plt.legend(('Precision','Recall'), loc='center right')
    #plt.savefig('Linear_Performance_' + model_name + '.png')
    plt.show()

plot_perf(DIG_sim_tp, 'True parents', axis_mean = None)
plot_perf(DIG_sim_corr, 'Most correlated assets', axis_mean = None)
plot_perf(DIG_sim_pf, 'Correlated Portfolio', axis_mean = None)
plot_perf(DIG_sim_pf_A, 'Ideal Portfolio', axis_mean = None)
#plot_perf(DIG_sim_rnn, 'RNN', axis_mean = None)

#====================== precision recall curve with sklearn ============================================================
from sklearn.metrics import precision_recall_curve

precision_tp, recall_tp, thresholds_tp = precision_recall_curve(np.reshape(1*(A.T!=0),-1), np.reshape(DIG_sim_tp,-1))
precision_corr, recall_corr, thresholds_corr = precision_recall_curve(np.reshape(1*(A.T!=0),-1), np.reshape(DIG_sim_corr,-1))
precision_pf, recall_pf, thresholds_pf = precision_recall_curve(np.reshape(1*(A.T!=0),-1), np.reshape(DIG_sim_pf,-1))
precision_pf_A, recall_pf_A, thresholds_pf_A = precision_recall_curve(np.reshape(1*(A.T!=0),-1), np.reshape(DIG_sim_pf_A,-1))
#precision_pf_rnn, recall_pf_rnn, thresholds_pf_rnn = precision_recall_curve(np.reshape(1*(A.T!=0),-1), np.reshape(DIG_sim_rnn,-1))
#precision_random, recall_random, thresholds_random = precision_recall_curve(np.reshape(1*(A.T!=0),-1), np.reshape(DIG_sim_random,-1))
#precision_mult, recall_mult, thresholds_mult = precision_recall_curve(np.reshape(1*(A.T!=0),-1), np.reshape(DIG_sim_mult,-1))

plt.plot(recall_tp, precision_tp, '-', color='blue')
plt.plot(recall_corr, precision_corr, '-', color='green')
plt.plot(recall_pf, precision_pf, '-', color='orange')
plt.plot(recall_pf_A, precision_pf_A, '-', color='black')
#plt.plot(recall_pf_rnn, precision_pf_rnn, '-', color='black')
#plt.plot(recall_random, precision_random, '-', color='red')
#plt.plot(recall_mult, precision_mult, '-', color='brown')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(('True Parents', 'Correlated Assets' , 'Portfolio', 'Ideal Portfolio'), loc='center right')
#plt.savefig('pre_rec_curves_gaussian.png')
plt.show()

#================================= distance between correlated and ideal portfolio when increase number of periods =====
# this section studies the error between the weights of the correlated portfolio and the true influence, as a function of
# the number of periods that we take to maximize correlation
nb_periods = 1000
nb_iter =100

x0 = np.random.randn(A.shape[0], 1)
X = np.zeros([1,A.shape[0]])
error0 = []
error1 = []


t0 = time.time()
for k in range(nb_iter):
    for i in range(1,nb_periods):
        eps =  np.random.randn(A.shape[0], 1)
        new_data = np.dot(A, X[-1].T ) + eps.T
        X = np.concatenate((X, new_data), axis=0)
    if k == 0:
        X = np.delete(X, 0, 0)

    X = pd.DataFrame(X)

    variables = X.iloc[:horizon, :].T.stack(dropna=False).reset_index(name='new')['new'].T
    for i in range(1, int(X.shape[0] / horizon)):
        variables = pd.concat([variables, X.iloc[i * horizon: i * horizon + horizon, :].T.stack(
            dropna=False).reset_index(name='new')['new']], axis=1)
    variables = variables.T.reset_index(drop=True)
    variables = np.asarray(variables)

    iter = list(range(A.shape[0]))

    firmlist = pd.DataFrame(data={'PERMCO': iter, 'COMNAM': iter, 'TICKER': iter})


    error0.append(la.norm((conditioning_with_one_pf(variables,X, A, 0, 2, firmlist , horizon)['comparison'][:,0]
                    - conditioning_with_one_pf(variables,X, A, 0, 2, firmlist , horizon)['comparison'][:,1])))
    error1.append(la.norm((conditioning_with_one_pf(variables,X, A, 0, 19, firmlist , horizon)['comparison'][:,0]
                    - conditioning_with_one_pf(variables,X, A, 0, 19, firmlist , horizon)['comparison'][:,1])))

    X = np.asarray(X)

    t1 = time.time()
    print('time elapsed : ', (t1 - t0) / 60, 'minutes', 'step:', k)

plt.style.use('seaborn-whitegrid')
plt.plot(list(range(nb_iter)), error0, '-', color='red')
plt.plot(list(range(nb_iter)), error1, '-', color='blue')
plt.title('Convergence in T of estimation errors')
plt.ylim([0, 0.4])
plt.xlabel('Nb of periods (in thousands)')
plt.ylabel('Distance between weights of projection and ideal projection')
plt.legend(('0', 'non 0'), loc='upper right')
plt.savefig('errors.png')
plt.show()

#============================================== save ===================================================================

with open( 'DIG_linear_knn.pkl', 'wb' ) as f:
    pickle.dump([DIG_sim_tp,DIG_sim_corr,DIG_sim_pf,DIG_sim_pf_A, DIG_sim_mult, X, A], f)

with open( 'DIG_linear_gaussian.pkl', 'wb' ) as f:
    pickle.dump([DIG_sim_tp,DIG_sim_corr,DIG_sim_pf,DIG_sim_pf_A, X, A], f)

