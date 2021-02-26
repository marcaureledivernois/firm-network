import numpy as np
from scipy.special import gamma, psi
from scipy.linalg import det
from numpy import pi
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import keras.backend as kb
import matplotlib.pyplot as plt

__all__=['entropy', 'mutual_information', 'entropy_gaussian']

EPS = np.finfo(float).eps


def nearest_distances(X, k=1):
    """
    X = array(N,M)
    N = number of points
    M = number of dimensions

    returns the distance to the kth nearest neighbor for every point in X
    """
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor


def entropy_gaussian(C):
    """
    Entropy of a gaussian variable with covariance matrix C
    """

    if np.isscalar(C): # C is the variance
        return .5*(1 + np.log(2*pi)) + .5*np.log(C)
    else:
        n = C.shape[0] # dimension
        return .5*n*(1 + np.log(2*pi)) + .5*np.log(abs(det(C)))


def entropy(X, k=1):
    """
    Returns the entropy of the X.
    Parameters
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed

    k : int, optional
        number of nearest neighbors for density estimation

    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    """

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k) #distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)  /2**d
    """
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.

    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    """
    return (d*np.mean(np.log(2*r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information(variables, k=1):
    """
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions

    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation

    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    """
    if len(variables) < 2:
        raise AttributeError(
                "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([entropy(X, k=k) for X in variables])
            - entropy(all_vars, k=k))


def directed_information(variables, T, k):
    """
    Returns the directed information from x to y causally condition on z.
    I (x --> y || z)
    Each variable is a matrix x = array(n_samples, n_features)

    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
      T = observation horizon

    Example: directed_information((X, Y, Z), k=5)
    """
    if len(variables) < 3:
        raise AttributeError(
                "Directed information must involve at least 3 variables")
    if T < 2:
        raise AttributeError(
                "observation horizon must be greater than 1")
    directed = 0
    for t in range(T-1):
        X = variables[0][:, 0:(t+1)]      # R_i,upto(t-1)
        Y1 = variables[1][:, 0:(t + 1)]   # R_j,upto(t-1)
        Y2 = variables[1][:, 0:(t + 2)]   # R_j,upto(t)
        if len(variables[2]) > 0:
            Z = variables[2][:, 0:(t + 1)] # R_\{i,j},upto(t-1)
            for _ in range(1, np.int(variables[2].shape[1] / T)):
                Z = np.concatenate((Z, variables[2][:, _ * T: _ * T + (t + 1)]), axis=1)

            directed += max(entropy(np.concatenate((Z, Y2), axis=1), k=k) + \
                        entropy(np.concatenate((Z, Y1, X), axis=1), k=k) \
                         - entropy(np.concatenate((Z, Y1), axis=1), k=k) \
                         - entropy(np.concatenate((Z, Y2, X), axis=1), k=k),0)

        else:
            directed +=  max(entropy(Y2, k=k) \
                         + entropy(np.concatenate((Y1, X), axis=1), k=k) \
                         - entropy(Y1, k=k) \
                         - entropy(np.concatenate((Y2, X), axis=1), k=k),0)
    return directed



def directed_information_gaussian(variables, time_horizon, i, j, set_k):
    """
    Computes the DI in a multi-variate Normal distribution
    with covariance matrix covariance of dimension (3*T)*(3*T) ( 3 ??????)
    """
    dI_th = 0

    x_i = variables[:, i * time_horizon:(i + 1) * time_horizon]
    x_j = variables[:, j * time_horizon:(j + 1) * time_horizon]
    ind = [0]
    for i_ in set_k:
        i_rest = np.array(range(i_ * time_horizon, (i_ + 1) * time_horizon))
        ind = np.concatenate((ind, i_rest), axis=0)
    ind = np.delete(ind, 0)
    x_set_k = variables[:, ind]

    covariance = np.cov(np.concatenate((x_i, x_j, x_set_k),axis = 1), rowvar=False)

    z = np.array(range(2 * time_horizon, covariance.shape[1], time_horizon), dtype=np.int16)
    for t in range(time_horizon - 1):
        x = np.array(range(0, t + 1),  dtype=np.int16)
        y1 = np.array(range(time_horizon, time_horizon + t + 1), dtype=np.int16)
        y2 = np.array(range(time_horizon, time_horizon + t + 2),  dtype=np.int16)
        if t>0:
            z = np.concatenate((z,np.array(range(2 * time_horizon, covariance.shape[1] ,time_horizon))+t))
        z.sort()
        ind1 = np.concatenate((y2, z), axis=0)
        ind2 = np.concatenate((x, y1, z), axis=0)
        ind3 = np.concatenate((y1, z), axis=0)
        ind4 = np.concatenate((x, y2, z), axis=0)
        dI_th += max(entropy_gaussian(covariance[np.ix_(ind1, ind1)])
                  + entropy_gaussian(covariance[np.ix_(ind2, ind2)])
                  - entropy_gaussian(covariance[np.ix_(ind3, ind3)])
                  - entropy_gaussian(covariance[np.ix_(ind4, ind4)]),0)
    return dI_th / time_horizon


def directed_information_gaussian_with_pf(variables, time_horizon, i, j, x_set_k):
    """
    Computes the DI in a multi-variate Normal distribution
    with covariance matrix covariance of dimension (3*T)*(3*T) ( 3 ??????)
    """
    dI_th = 0

    x_i = variables[:, i * time_horizon:(i + 1) * time_horizon]
    x_j = variables[:, j * time_horizon:(j + 1) * time_horizon]

    covariance = np.cov(np.concatenate((x_i, x_j, x_set_k),axis = 1), rowvar=False)
    z = np.array(range(2 * time_horizon, covariance.shape[1], time_horizon), dtype=np.int16)
    for t in range(time_horizon - 1):
        x = np.array(range(0, t + 1), dtype=np.int16)
        y1 = np.array(range(time_horizon, time_horizon + t + 1), dtype=np.int16)
        y2 = np.array(range(time_horizon, time_horizon + t + 2), dtype=np.int16)
        if t > 0:
            z = np.concatenate((z, np.array(range(2 * time_horizon, covariance.shape[1], time_horizon)) + t))
        z.sort()
        ind1 = np.concatenate((y2, z), axis=0)
        ind2 = np.concatenate((x, y1, z), axis=0)
        ind3 = np.concatenate((y1, z), axis=0)
        ind4 = np.concatenate((x, y2, z), axis=0)
        dI_th += entropy_gaussian(covariance[np.ix_(ind1, ind1)]) \
                  + entropy_gaussian(covariance[np.ix_(ind2, ind2)]) \
                  - entropy_gaussian(covariance[np.ix_(ind3, ind3)]) \
                  - entropy_gaussian(covariance[np.ix_(ind4, ind4)])
    return dI_th / time_horizon



def conditioning_correl(data, firmlist, i, j , nb_highest_corr = 10):
    """
    Computes the indices of the condition variables K_{i,j} in the Directed Information Graph. These are the
    institutions with highest 1 period lagged-correlation with Rj (except Ri and Rj).
    :param corrmat: correlation matrix of the time series. shape : [Nb days, Nb firms]
    :param i: node i
    :param j: node j
    :param nb_highest_corr: number of highest correlations to take into account
    :return: x_set_k
    """

    mask = ~(data.columns.isin(firmlist['PERMCO'][[j]]))
    cols_to_shift = data.columns[mask]
    data = data.copy()
    data[cols_to_shift] = data.loc[:, mask].shift(1)

    corrmat = data.corr()
    corrmat = corrmat.reset_index(drop=True)
    corrmat = corrmat.T.reset_index(drop=True).T

    return corrmat.drop([i,j]).sort_values(j, ascending=False)[:nb_highest_corr].index.tolist()

def directed_information_in_network(variables, time_horizon, i, j, set_k):
    """
    Computes the DI in a network of variables
    from node i to j given the rest of the network, i.e.,
    I( x_i --> x_j || x_set_k)
    """
    x_i = variables[:, i * time_horizon:(i + 1) * time_horizon]
    x_j = variables[:, j * time_horizon:(j + 1) * time_horizon]
    ind = [0]
    for i_ in set_k:
        i_rest = np.array(range(i_ * time_horizon, (i_ + 1) * time_horizon))
        ind = np.concatenate((ind, i_rest), axis=0)
    ind = np.delete(ind, 0)
    x_set_k = variables[:, ind]

    return directed_information((x_i, x_j, x_set_k), T=time_horizon, k=1)

def directed_information_in_network_corr_and_pf(variables, time_horizon, i, j, set_k, x_set_k_):
    """
    Computes the DI in a network of variables
    from node i to j given the rest of the network, i.e.,
    I( x_i --> x_j || x_set_k)
    """
    x_i = variables[:, i * time_horizon:(i + 1) * time_horizon]
    x_j = variables[:, j * time_horizon:(j + 1) * time_horizon]
    ind = [0]
    for i_ in set_k:
        i_rest = np.array(range(i_ * time_horizon, (i_ + 1) * time_horizon))
        ind = np.concatenate((ind, i_rest), axis=0)
    ind = np.delete(ind, 0)
    x_set_k = variables[:, ind]

    x_set_k = np.hstack((x_set_k,x_set_k_))
    return directed_information((x_i, x_j, x_set_k), T=time_horizon, k=1)

def series_to_supervised(data,i,j,firmlist):
    df = data.copy()
    ts_to_shift = df.loc[:, firmlist['PERMCO'][[j]]].shift(-1)
    df = df.drop(columns=firmlist['PERMCO'][[i]])
    df = pd.concat((df,ts_to_shift),axis=1)
    df.dropna(inplace=True)
    return df

def conditioning_with_rnn(X,i,j,firmlist,horizon):
    reframed = series_to_supervised(X, i, j, firmlist)

    # split into train and test sets
    values = reframed.values
    train_size = int(0.85 * values.shape[0])
    train = values[:train_size, :]
    test = values[train_size:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # design network
    kb.clear_session()
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.5))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(40, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(30))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=160, batch_size=72, validation_data=(test_X, test_y), verbose=0,
                        shuffle=False)

    res = dict()
    P = model.predict(np.concatenate([train_X, test_X], axis=0))
    res['x_set_k'] = np.reshape(P[:int(P.shape[0] / horizon) * horizon], (-1, horizon))
    #res['correlation'] = pd.concat([X[firmlist['PERMCO'][j]].reset_index(drop=True), pd.DataFrame(P).shift(1)], axis=1).corr().iloc[0,1]
    return res

def max_corr_pf(sig_xr,sig_rr):
    w = np.dot(np.linalg.inv(sig_rr), np.reshape(sig_xr, (sig_xr.shape[0], 1))) #/ \
         #np.sqrt(np.dot(np.dot(np.reshape(sig_xr, (1, sig_xr.shape[0])), np.linalg.inv(sig_rr)),
         #               np.reshape(sig_xr, (sig_xr.shape[0], 1))))
    return w

def conditioning_with_one_pf(variables, data_T, A, i, j, firmlist , horizon):
    x_j = variables[:, j * horizon:(j + 1) * horizon]
    x_minus_i = np.delete(variables, list(range(i * horizon, (i + 1) * horizon )),1)

    t = 3
    x_jt = x_j[:,t]
    x_minus_i_tminus1 = x_minus_i[:,list(range(t-1,x_minus_i.shape[1],horizon))]

    sigma = np.cov(np.concatenate((np.reshape(x_jt,(x_minus_i_tminus1.shape[0],1)), x_minus_i_tminus1),axis = 1), rowvar=False)

    sig_xr = sigma[0,1:]
    sig_rr = sigma[1:,1:]

    res = dict()
    res['w_k'] = max_corr_pf(sig_xr, sig_rr)
    #res['w_k'] = np.where(np.abs(res['w_k'])<0.03,0,res['w_k'])

    #res['comparison'] = np.concatenate((res['w_k'],np.reshape(np.delete(A,i,1)[j,:],res['w_k'].shape)) ,axis=1)
    pf_return = np.dot(data_T.drop(columns=firmlist['PERMCO'][[i]]), res['w_k'])


    res['x_set_k'] = np.reshape(pf_return[:int(pf_return.shape[0] / horizon)*horizon], (-1, horizon))
    res['correlation'] = pd.concat([data_T[firmlist['PERMCO'][j]].reset_index(drop=True), pd.DataFrame(pf_return).shift(1)], axis=1).corr().iloc[0,1]
    return res

def conditioning_with_A(data_T, A, i, j, firmlist , horizon):

    res = dict()
    weights = np.delete(A[j,:],i,0)

    pf_return = np.dot(data_T.drop(columns=firmlist['PERMCO'][[i]]), weights)
    res['w_k'] = weights
    res['x_set_k'] = np.reshape(pf_return[:int(pf_return.shape[0] / horizon)*horizon], (-1, 5))
    res['correlation'] = pd.concat([data_T[firmlist['PERMCO'][j]].reset_index(drop=True), pd.DataFrame(pf_return).shift(1)], axis=1).corr().iloc[0,1]
    return res

def gaussian_basis(x,xi,gam):
    return np.exp(-gam*(np.sum((x- np.dot(np.ones((x.shape[0],1)),np.reshape(xi,(1,xi.shape[0]))))**2,axis=1)))

def logmatrix(B,K):
    logB = np.zeros(B.shape)
    for k in range(1,K):
        logB = logB + ((-1)**(k+1)) * np.linalg.matrix_power((B - np.eye(B.shape[0])),k) / k
    return logB

def conditioning_with_koopman(i,j,beg, end, X,firmlist, horizon):
    gamma = 0.01

    Px = np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=np.delete(beg, i, axis=1), gam=gamma, xi=0)
    for xi in np.linspace(X.min().min(), X.max().max(), 1000):
        Px = np.vstack(
            [Px, np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=np.delete(beg, i, axis=1), gam=gamma, xi=xi)])
    Px = np.delete(Px, 0, 0)
    Px = Px.T

    Py = np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=np.delete(end, i, axis=1), gam=gamma, xi=0)
    for xi in np.linspace(X.min().min(), X.max().max(), 1000):
        Py = np.vstack(
            [Py, np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=np.delete(end, i, axis=1), gam=gamma, xi=xi)])
    Py = np.delete(Py, 0, 0)
    Py = Py.T

    L = logmatrix(np.dot(Py, np.linalg.pinv(Px)), 20) / horizon

    Fhat = np.dot(L, beg)

    Hx = Px  # because bases function = library functions

    parameters = {'alpha': [1e-4, 0.1, 1]}

    lasso_regressor = GridSearchCV(Lasso(), parameters, scoring='neg_mean_squared_error', cv=5)
    lasso_regressor.fit(Hx, Fhat[:, j])
    lasso = Lasso(alpha= lasso_regressor.best_params_['alpha']).fit(Hx, Fhat[:, j])
    wj = lasso.coef_

    hX = np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=X.drop(columns=firmlist['PERMCO'][[i]]), gam=gamma,
                             xi=0)
    for xi in np.linspace(X.min().min(), X.max().max(), 1000):
        hX = np.vstack([hX,
                        np.apply_along_axis(func1d=gaussian_basis, axis=1, arr=X.drop(columns=firmlist['PERMCO'][[i]]),
                                            gam=gamma, xi=xi)])
    hX = np.delete(hX, 0, 0)
    hX = hX.T

    P = np.dot(hX, wj)
    P = np.reshape(P[:int(P.shape[0] / horizon) * horizon], (-1, 5))
    return P


def conditioning_with_trueparents(A, i, j):
    set_k = [par_j for par_j in range(A.shape[0]) if A[j,par_j] != 0 and par_j not in [i,j]]
    return set_k


def conditioning_with_random_assets(i, j, nb_assets, nb_random_assets=10):
    rnd = random.sample(range(nb_assets), nb_random_assets)
    set_k = [item for item in rnd if item not in [i,j]]
    return set_k


def directed_information_in_network_with_pf(variables, time_horizon, i, j, x_set_k):
    """
    Computes the DI in a network of variables
    from node i to j given the rest of the network, i.e.,
    I( x_i --> x_j || x_set_k)
    """
    x_i = variables[:x_set_k.shape[0], i * time_horizon:(i + 1) * time_horizon]
    x_j = variables[:x_set_k.shape[0], j * time_horizon:(j + 1) * time_horizon]
    return directed_information((x_i, x_j, x_set_k), T=time_horizon, k=1)



def build_another_pf(sigma,i,j,with_i_idx_toremove, without_i_idx_toremove, data_T,A,firmlist,horizon):

    sigma = np.delete(sigma,[1+x for x in without_i_idx_toremove],0)
    sigma = np.delete(sigma, [1+x for x in without_i_idx_toremove], 1)

    sig_xr = sigma[0,1:]
    sig_rr = sigma[1:,1:]

    res_pf = dict()
    w_k = max_corr_pf(sig_xr, sig_rr)
    w_k = np.where(np.abs(w_k) < 0.03, 0, w_k)
    res_pf['w_k'] = w_k
    res_pf['comparison'] = np.concatenate((w_k,np.reshape(np.delete(A,[i] + with_i_idx_toremove,1)[j,:],w_k.shape)) ,axis=1)
    res_pf['pf_return'] = np.dot(data_T.drop(columns=firmlist['PERMCO'][[i] + with_i_idx_toremove]), w_k)
    res_pf['correlation'] = pd.concat([data_T[firmlist['PERMCO'][j]].reset_index(drop=True), pd.DataFrame(res_pf['pf_return']).shift(1)], axis=1).corr().iloc[0,1]
    res_pf['highest_abs_weight'] = abs(w_k).argmax()
    res_pf['x_set_k'] = np.reshape(res_pf['pf_return'][:int(res_pf['pf_return'].shape[0] / horizon) * horizon], (-1, horizon))
    return res_pf


def conditioning_with_multiple_pf(variables, data_T, A, i, j, firmlist , horizon, nb_add_pf):
    """
    :return:
    comparison: compare w_k from projection and row j of matrix A
    compare_wk : compare w_k of consecutive projections
    """
    x_j = variables[:, j * horizon:(j + 1) * horizon]
    x_minus_i = np.delete(variables, list(range(i * horizon, (i + 1) * horizon )),1)

    t = 3
    x_jt = x_j[:,t]
    x_minus_i_tminus1 = x_minus_i[:,list(range(t-1,x_minus_i.shape[1],horizon))]

    sigma = np.cov(np.concatenate((np.reshape(x_jt,(x_minus_i_tminus1.shape[0],1)), x_minus_i_tminus1),axis = 1), rowvar=False)

    sig_xr = sigma[0,1:]
    sig_rr = sigma[1:,1:]

    res = dict()
    res['comparison'] = []
    res['w_k'] = []
    res['correlation'] = []
    res['highest_abs_weight'] = []

    w_k = max_corr_pf(sig_xr, sig_rr)
    w_k = np.where(np.abs(w_k) < 0.03, 0, w_k)
    res['w_k'].append(w_k)
    res['compare_wk'] = w_k
    res['comparison'].append(np.concatenate((w_k,np.reshape(np.delete(A,i,1)[j,:],w_k.shape)) ,axis=1))

    pf_return = np.dot(data_T.drop(columns=firmlist['PERMCO'][[i]]), w_k)
    res['pf_return'] = pf_return

    res['correlation'].append(pd.concat([data_T[firmlist['PERMCO'][j]].reset_index(drop=True), pd.DataFrame(pf_return).shift(1)], axis=1).corr().iloc[0,1])
    res['x_set_k'] = np.reshape(res['pf_return'][:int(res['pf_return'].shape[0] / horizon) * horizon], (-1, horizon))

    highest_abs_weight = np.abs(w_k).argmax()
    res['highest_abs_weight'].append(highest_abs_weight)

    full_with_i = list(range(data_T.shape[1]))
    full_with_i.pop(i)
    with_i_idx_toremove = []
    with_i_idx_toremove.append(full_with_i.pop(highest_abs_weight))

    full = list(range(w_k.shape[0]))
    true_indices = []
    true_indices.append(full.pop(highest_abs_weight))

    for pf in range(nb_add_pf):
        res_pf = build_another_pf(sigma, i, j, with_i_idx_toremove, true_indices , data_T, A, firmlist,horizon)
        res['comparison'].append(res_pf['comparison'])
        #w_k_with_na = np.insert(res_pf['w_k'],true_indices,np.nan)
        #w_k_reshaped = np.reshape(w_k_with_na,(A.shape[0]-1,1))
        #res['compare_wk'] = np.concatenate((res['compare_wk'], w_k_reshaped ),axis = 1)
        res['w_k'].append(res_pf['w_k'])
        res['pf_return'] = np.concatenate((res['pf_return'],res_pf['pf_return']),axis=1)
        res['correlation'].append(res_pf['correlation'])
        res['x_set_k'] = np.concatenate((res['x_set_k'], res_pf['x_set_k']), axis = 1)
        highest_abs_weight = np.abs(res_pf['w_k']).argmax()
        res['highest_abs_weight'].append(highest_abs_weight)
        with_i_idx_toremove.append(full_with_i.pop(highest_abs_weight))
        true_indices.append(full.pop(highest_abs_weight))
    return res

