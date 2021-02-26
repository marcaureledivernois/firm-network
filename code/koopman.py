import numpy as np
from mutualinfo import entropy, entropy_gaussian, mutual_information, directed_information,\
    directed_information_gaussian, directed_information_in_network, conditioning_correl, \
    directed_information_in_network_with_pf,  max_corr_pf,\
    conditioning_with_one_pf, conditioning_with_trueparents, directed_information_in_network_corr_and_pf, conditioning_with_koopman
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
import scipy.linalg as la
import pickle
import random
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

nb_assets = 30
nb_parents = 20
nb_periods = 10000
A = np.zeros([nb_assets,nb_assets,nb_assets]) #elements A[x,y,i]+A[y,x,i] represent interaction effect of x and y in the process i
epsilon = 0.6

A_tp = np.zeros([nb_assets,nb_assets])

for i in range(A.shape[2]):
    start = random.sample(range(0, nb_assets), nb_parents)
    A[start, : ,i] = np.random.uniform(low=-0.9, high=0.9, size=(A[start, : ,i].shape)) #generate influence
    A[:, start, i] = np.random.uniform(low=-0.9, high=0.9, size=(A[:, start, i].shape))  # generate influence
    lowinf = np.abs(np.sum((A[:,:,i] + A[:,:,i].T) ,axis=1)) < 1*np.mean(np.abs(np.sum((A[:,:,i] + A[:,:,i].T) ,axis=1)))
    A[lowinf,:,i] = 0
    A[:, lowinf, i] = 0
    #A[:,:,i] = A[:,:,i] / (max(abs(la.eig(A[:,:,i])[0])) + 0.1)  #stationarity of A
    A_tp[i,:] = 1*(np.sum((A[:,:,i] + A[:,:,i].T) ,axis=1) != 0)

b = np.ones([nb_assets]) * 0.7

X = np.ones([nb_periods,nb_assets])
for t in range(1,nb_periods):
    for i in range(nb_assets):
            pX = np.power(np.abs(X[t-1,:]),1/3)*np.sign(X[t-1,:])
            X[t,i] = b[i] * np.dot(pX, np.dot(A[:,:,i], pX.T )) + 0.2 * np.random.randn() + 0.00 * X[t-1,i]
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

#===================================== BUILD KOOPMAN MATRICES  =================================================================

D = np.asarray(X.iloc[:horizon,:]).flatten()
for i in range(1, int(X.shape[0] / horizon)):
    D = np.vstack([D, np.asarray(X.iloc[i * horizon : i * horizon + horizon,:]).flatten()])


beg = D[:,:nb_assets]
end = D[:,-nb_assets:]


def gaussian_basis(x,xi,gam):
    return np.exp(-gam*(np.sum((x- np.dot(np.ones((x.shape[0],1)),np.reshape(xi,(1,xi.shape[0]))))**2,axis=1)))

def monomials_upto_order2(x):
    all = np.dot(np.reshape(x, (nb_assets - 1, 1)),np.reshape(x, (1,nb_assets - 1)))
    triu = all[np.triu_indices(nb_assets - 1)]
    return np.concatenate((triu,x))

def logmatrix(B,K):
    logB = np.zeros(B.shape)
    for k in range(1,K):
        logB = logB + ((-1)**(k+1)) * np.linalg.matrix_power((B - np.eye(B.shape[0])),k) / k
    return logB

DIG_sim_koopman = np.zeros((nb_assets,nb_assets))

for i in range(nb_assets):

    # gaussian radial basis
    gamma = 0.01

    Px = gaussian_basis(np.delete(beg, i, axis=1),np.delete(beg, i, axis=1)[0,:],gamma)
    Py = gaussian_basis(np.delete(end, i, axis=1), np.delete(end, i, axis=1)[0, :], gamma)
    for k in range(beg.shape[0]):
        Px = np.vstack([Px, gaussian_basis(np.delete(beg, i, axis=1),np.delete(beg, i, axis=1)[k,:],gamma)])
        Py = np.vstack([Py, gaussian_basis(np.delete(end, i, axis=1), np.delete(beg, i, axis=1)[k, :], gamma)])
    Px = np.delete(Px, 0, 0)
    Px = Px.T
    Py = np.delete(Py, 0, 0)
    Py = Py.T



    Px = np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=np.delete(beg, i, axis=1), gam=gamma, xi=np.delete(beg, i, axis=1)[0,:])
    for k in range(beg.shape[0]):
        Px = np.vstack(
            [Px, np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=np.delete(beg, i, axis=1), gam=gamma, xi=np.delete(beg, i, axis=1)[k,:])])
    Px = np.delete(Px, 0, 0)
    Px = Px.T

    Py = np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=np.delete(end, i, axis=1), gam=gamma, xi=0)
    for xi in np.linspace(X.min().min()*2, X.max().max()*2, 10000):
        Py = np.vstack(
            [Py, np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=np.delete(end, i, axis=1), gam=gamma, xi=xi)])
    Py = np.delete(Py, 0, 0)
    Py = Py.T

    hX = np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=X.drop(columns=firmlist['PERMCO'][[i]]), gam=gamma, xi=0)
    for xi in np.linspace(X.min().min(), X.max().max(), 1000):
        hX = np.vstack([hX,
                        np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=X.drop(columns=firmlist['PERMCO'][[i]]),
                                            gam=gamma, xi=xi)])
    hX = np.delete(hX, 0, 0)
    hX = hX.T

    #monomials of order 2 basis functions
    #Px = np.apply_along_axis(func1d=monomials_upto_order2, axis=1, arr=np.delete(beg, i, axis=1))
    #Py = np.apply_along_axis(func1d=monomials_upto_order2, axis=1, arr=np.delete(end, i, axis=1))

    #BB = np.dot(Py, np.linalg.pinv(Px))
    #norm = np.linalg.norm(BB-np.eye(BB.shape[0]),2)

    #hX = np.apply_along_axis(func1d=monomials_upto_order2, axis=1, arr=X.drop(columns=firmlist['PERMCO'][[i]]))

    #cont'd

    #L = logmatrix(np.dot(Py, np.linalg.pinv(Px)), 20) / horizon
    K = np.dot(Py, np.linalg.inv(Px))
    L_scipy = la.logm(K) / horizon

    Fhat = np.dot(L_scipy, beg)

    Hx = Px  # because bases function = library functions

    parameters = {'alpha': [1e-4, 0.1, 1]}

    for j in range(nb_assets):
        lasso_regressor = GridSearchCV(Lasso(), parameters, scoring='neg_mean_squared_error', cv=5)
        lasso_regressor.fit(Hx, Fhat[:, j])
        lasso = Lasso(alpha= lasso_regressor.best_params_['alpha']).fit(Hx, Fhat[:, j])
        wj = lasso.coef_

        wj = np.dot(np.linalg.pinv(Hx), Fhat[:, j])

        P = np.dot(hX, wj)
        P = np.reshape(P[:int(P.shape[0] / horizon) * horizon], (-1, 5))

        DIG_sim_koopman[i,j] = directed_information_in_network_with_pf(variables, horizon, i, j, P)
        print(str(i) + ' ' + str(j))

#============================================= DIG_binary ==============================================================

def DIG_binary(DIG, threshold, axis=None):
    if axis==None:
        DIG_b = (DIG > threshold)*1
    else:
        DIG_b = (DIG > threshold) * 1
    return DIG_b


DIG_binary_tp = DIG_binary(DIG=DIG_sim_koopman, threshold = 0, axis = None)


#====================== precision recall curve with sklearn ============================================================
from sklearn.metrics import precision_recall_curve

precision_k, recall_k, thresholds_k = precision_recall_curve(np.reshape(1*(A_tp.T!=0),-1), np.reshape(DIG_sim_koopman,-1))

plt.plot(recall_k, precision_k, '-', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(('Koopman'), loc='upper right')
#plt.savefig('pre_rec_curves_gaussian.png')
plt.show()