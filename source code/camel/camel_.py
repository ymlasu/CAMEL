# This is source code for Curvature Augmeented Manifold Emebedding and Learning -- CAMEL

# It is used for dimension reduction and data visulization of high dimention data. Detailed instruction can be found at

# https://github.com/ymlasu/CAMEL

# Author: Yongming Liu

# Email: yongming.liu@asu.edu

# License: MIT License

# Initia release date: 02/2024




import numba
import time
import math
import datetime
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import preprocessing

from annoy import AnnoyIndex

global _RANDOM_STATE
_RANDOM_STATE = None


@numba.njit("f4[:,:](f4[:,:],i4[:,:])", parallel=True, nogil=True, cache=True, fastmath=True)
def avg_coor_compu(Xi=np.array([]), indices=np.array([])):
    """
    Compute the average coordinate for each node, use in the curvature computing
    
    Xi: ndarray
    - input high dim or low dim coordinates
    
    indices: nd array
    - nbrs neighbor list
    """
    n, num_neighbors = indices.shape
    dim=Xi.shape[1]
    
    avg_coor=np.zeros((n, dim), dtype=np.float32)

    for i in numba.prange(n):
          
        for j in range (num_neighbors):
            avg_coor[i, :] += Xi [indices[i,j],:]
    avg_coor= avg_coor/num_neighbors
    
    return avg_coor

@numba.njit("f4[:](f4[:,:],f4[:,:],i4[:,:])", parallel=True, nogil=True, cache=True, fastmath=True)
def curv_nb(Xi=np.array([]), avg_coor=np.array([]), indices=np.array([])):
    """
    Compute the curvature of neighbors in low dimention and high dimention
    Xi: ndarray
    - high dim or low dim coordinates
    
    avg_coor: ndarray
    - average distance between node neighbor points, calculated outside to save computation time
    
    indcies: ndarray
    - nbrs, neighbor list
    """
    n, num_neighbors = indices.shape
    Edge_Curvature=np.zeros((n*num_neighbors), dtype=np.float32)
    distances_pair=np.zeros((n,num_neighbors), dtype=np.float32)
    distances_multi=np.zeros((n,num_neighbors), dtype=np.float32)
    dim=Xi.shape[1]

    for i in numba.prange(n):
        for j in range (num_neighbors):
            node_starting=i
            node_ending=indices[i,j]
            x1=Xi[node_starting,:]
            x2=Xi[node_ending,:]
            x11=avg_coor[node_starting,:]
            x22=avg_coor[node_ending,:]
            for k in numba.prange (dim):
                distances_pair[i,j] += (x1[k] - x2[k]) ** 2
                distances_multi[i,j] += (x11[k] - x22[k]) ** 2
            distances_pair[i,j]=(distances_pair[i,j])**0.5    
            distances_multi[i,j]=(distances_multi[i,j])**0.5   

            Edge_Curvature[i*num_neighbors + j]=(1.0-distances_multi[i,j]/max(distances_pair[i,j],0.001))
    
    return Edge_Curvature


@numba.njit("f4(f4[:],f4[:])", cache=True)
def euclid_dist(x1, x2):
    """
    Euclidean distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return np.sqrt(result)

def target_scale_compute_1D(X, target, number_samples):
    """
    Used to comupute the scaling factor for labels/targets during the supervised learning
    Parameters, this is applicable to 1D target vector
    ----------
    X : ndarray, float32
        feature cordinates
        
    target : ndarray, float32
        target/labels
        
    number_samples : int
        number of samples to estimate the avrage distance among features and targets

    Returns
    -------
    scale_factor : float32
        scale factor multiply the targets/labels to make the 
        distance of features and distances of targets almost equal

    """
    Xmaximum, Xdim = X.shape
    Targetmaximum = target.shape[0]
    X_distance=0.0
    Target_distance=0.0

    for i in numba.prange(number_samples):
        sampleX1=np.random.randint(Xmaximum)
        sampleX2=np.random.randint(Xmaximum)
        sampleTarget1=np.random.randint(Targetmaximum)
        sampleTarget2=np.random.randint(Targetmaximum)
        X_distance += np.linalg.norm(X[sampleX1,:] - X[sampleX2, :])
        Target_distance += np.linalg.norm(target[sampleTarget1] - target[sampleTarget2])    
    
    scale_factor= X_distance/max(Target_distance, 1e-6)
    
    return scale_factor   
    
def target_scale_compute_2D(X, target, number_samples):
    """
    Used to comupute the scaling factor for labels/targets during the supervised learning
    Parameters, this is applicable to 2D target matrix
    ----------
    X : ndarray, float32
        feature cordinates
        
    target : ndarray, float32
        target/labels
        
    number_samples : int
        number of samples to estimate the avrage distance among features and targets

    Returns
    -------
    scale_factor : float32
        scale factor multiply the targets/labels to make the 
        distance of features and distances of targets almost equal

    """
    Xmaximum, Xdim = X.shape
    Targetmaximum = target.shape[0]
    X_distance=0.0
    Target_distance=0.0
    for i in numba.prange(number_samples):
        sampleX1=np.random.randint(Xmaximum)
        sampleX2=np.random.randint(Xmaximum)
        sampleTarget1=np.random.randint(Targetmaximum)
        sampleTarget2=np.random.randint(Targetmaximum)
        X_distance += np.linalg.norm(X[sampleX1,:] - X[sampleX2, :])
        Target_distance += np.linalg.norm(target[sampleTarget1,:] - target[sampleTarget2, :])

    scale_factor= X_distance/max(Target_distance, 1e-6)
    
    return scale_factor   
    
    
@numba.njit("f4[:,:](f4[:,:],i4[:,:])", nogil=True, cache=True)
def init_transform_embedding(embedding,nbrs):
    """
    This is used to do the initialization of the transform operation, 
    can be for high dimentional or low dimentional

    Parameters
    ----------
    embedding : ndarray, float32
        This is the embedding from the fit or fit_transform. 
        - for feature learning case, embedding is the low-dim representaion
        - for inverse_trasnform, embedding is acturally the original feature space matrix
    nbrs : int
        - neighbor list.

    Returns
    -------
    Y_init : ndarray, float32
        initial embedding point, it should be noticed that only the points 
        beyond the original n_old embedding are generated using simple average of its neighbor points

    """
    n_total, n_neighbors = nbrs.shape
    n_old, Xdim = embedding.shape
    n_new=n_total-n_old
    Y_init=np.zeros((n_new,Xdim), dtype=np.float32)

    for i in range(n_new):
        for j in range(n_neighbors):            
            for d in range(Xdim):
                Y_init[i,d] += embedding[nbrs[n_old+i,j],d]

    Y_init=Y_init/n_neighbors
                    
    return Y_init

@numba.njit("i4[:](i4,i4,i4[:])", nogil=True, cache=True)
def sample_FP(n_samples, maximum, reject_ind):
    result = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(maximum)
            for k in range(i):
                if j == result[k]:
                    break
            for k in range(reject_ind.shape[0]):
                if j == reject_ind[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result


@numba.njit("i4[:,:](f4[:,:], i4[:,:],i4)", parallel=True, nogil=True, cache=True)
def sample_neighbors_pair(X, nbrs, n_neighbors):
    n = X.shape[0]
    pair_neighbors = np.empty((n*n_neighbors, 2), dtype=np.int32)

    for i in numba.prange(n):
        for j in numba.prange(n_neighbors):
            pair_neighbors[i*n_neighbors + j][0] = i
            pair_neighbors[i*n_neighbors + j][1] = nbrs[i][j]
    
    return pair_neighbors

@numba.njit("f4[:,:](f4[:,:], f4[:,:],i4)", parallel=True, nogil=True, cache=True)
def weight_neighbors_pair(X, distances, n_neighbors):
    n = X.shape[0]
    weight_neighbors = np.empty((n*n_neighbors,2), dtype=np.float32)
    #average distance of neighbor
    avg_distances=np.mean(distances)
    for i in numba.prange(n):
        for j in numba.prange(n_neighbors):
            weight_neighbors[i*n_neighbors + j][0] = 1.0-1.0*(np.arctan(1*(distances[i][j]/avg_distances-1.0))/np.pi)
            weight_neighbors[i*n_neighbors + j][1] = avg_distances/distances[i][j]
    
    return weight_neighbors

@numba.njit("i4[:,:](i4,f4[:,:],i4[:,:],i4)", parallel=True, nogil=True, cache=True)
def sample_neighbors_pair_basis(n_basis, X, nbrs, n_neighbors):
    '''Sample Nearest Neighbor pairs for new data.'''
    n = X.shape[0]
    pair_neighbors = np.empty((n*n_neighbors, 2), dtype=np.int32)

    for i in numba.prange(n):
        for j in numba.prange(n_neighbors):
            pair_neighbors[i*n_neighbors + j][0] = n_basis + i
            pair_neighbors[i*n_neighbors + j][1] = nbrs[i][j]
    return pair_neighbors

@numba.njit("f4[:,:](f4[:,:], f4[:,:],i4)", parallel=True, nogil=True, cache=True)
def weight_neighbors_pair_basis(X, distances, n_neighbors):
    n = X.shape[0]
    n_basis=distances.shape[0]-n
    weight_neighbors = np.empty((n*n_neighbors,2), dtype=np.float32)
    #average distance of neighbor
    avg_distances=np.mean(distances[:n_basis,:])
    for i in numba.prange(n):
        for j in numba.prange(n_neighbors):
            weight_neighbors[i*n_neighbors + j][0] = 1.0-1.0*(np.arctan(1*(distances[n_basis+i][j]/avg_distances-1.0))/np.pi)
            weight_neighbors[i*n_neighbors + j][1] = avg_distances/distances[n_basis+i][j]
    
    return weight_neighbors

@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4)", parallel=True, nogil=True, cache=True)
def sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP):
    '''Sample Further pairs.'''
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            FP_index = sample_FP(
                n_FP, n, pair_neighbors[i*n_neighbors: i*n_neighbors + n_neighbors][1])
            pair_FP[i*n_FP + k][0] = i
            pair_FP[i*n_FP + k][1] = FP_index[k]
    return pair_FP

@numba.njit("f4[:,:](f4[:,:], i4[:,:])", parallel=True, nogil=True, cache=True)
def weight_FP_pair(X, pair_FP):
    n=pair_FP.shape[0]
    weight_FP = np.empty((n,2), dtype=np.float32)
    #average distance of neighbor

    for t in numba.prange(n):

        weight_FP[t][0] = euclid_dist(X[pair_FP[t,0],:], X[pair_FP[t,1],:])
        weight_FP[t][1] = euclid_dist(X[pair_FP[t,0],:], X[pair_FP[t,1],:])
        
    avg_distances_FP=np.mean(weight_FP[:,0])
    weight_FP[:,1]=1.0+1.0*(np.arctan(1*(weight_FP[:,0]/avg_distances_FP-1.0))/np.pi)
    
    return weight_FP

@numba.njit("i4[:,:](i4, f4[:,:],i4[:,:],i4,i4)", parallel=True, nogil=True, cache=True)
def sample_FP_pair_basis(n_basis, X, pair_neighbors, n_neighbors, n_FP):
    '''Sample Further pairs for new data points.'''
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            FP_index = sample_FP(
                n_FP, n_basis, pair_neighbors[i*n_neighbors: i*n_neighbors + n_neighbors][1])
            pair_FP[i*n_FP + k][0] = n_basis + i
            pair_FP[i*n_FP + k][1] = FP_index[k]
    return pair_FP

@numba.njit("f4[:,:](f4[:,:], f4[:,:], i4[:,:])", parallel=True, nogil=True, cache=True)
def weight_FP_pair_basis(X, basis, pair_FP):
    n=pair_FP.shape[0]
    weight_FP = np.empty((n,2), dtype=np.float32)
    n_basis=basis.shape[0]
    #average distance of neighbor

    for t in numba.prange(n):

        weight_FP[t][0] = euclid_dist(X[pair_FP[t,0]-n_basis,:], basis[pair_FP[t,1],:])
        weight_FP[t][1] = euclid_dist(X[pair_FP[t,0]-n_basis,:], basis[pair_FP[t,1],:])
        
    avg_distances_FP=np.mean(weight_FP[:,0])
    weight_FP[:,1]=1.0+1.0*(np.arctan(1*(weight_FP[:,0]/avg_distances_FP-1.0))/np.pi)
    
    return weight_FP

@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4,i4)", parallel=True, nogil=True, cache=True)
def sample_FP_pair_deterministic(X, pair_neighbors, n_neighbors, n_FP, random_state):
    '''Sample Further pairs using the given random state.'''
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            np.random.seed(random_state+i*n_FP+k)
            FP_index = sample_FP(
                n_FP, n, pair_neighbors[i*n_neighbors: i*n_neighbors + n_neighbors][1])
            pair_FP[i*n_FP + k][0] = i
            pair_FP[i*n_FP + k][1] = FP_index[k]
    return pair_FP

@numba.njit("i4[:,:](i4,f4[:,:],i4[:,:],i4,i4,i4)", parallel=True, nogil=True, cache=True)
def sample_FP_pair_deterministic_basis(n_basis, X, pair_neighbors, n_neighbors, n_FP, random_state):
    '''Sample Further pairs using the given random state.'''
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            np.random.seed(random_state+i*n_FP+k)
            FP_index = sample_FP(
                n_FP, n, pair_neighbors[i*n_neighbors: i*n_neighbors + n_neighbors][1])
            pair_FP[i*n_FP + k][0] = n_basis+i
            pair_FP[i*n_FP + k][1] = FP_index[k]
    return pair_FP


@numba.njit("void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,f4,f4,i4)", parallel=True, nogil=True, cache=True)
def update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr):
    '''Update the embedding with the gradient'''
    n, dim = Y.shape
    lr_t = lr * math.sqrt(1.0 - beta2**(itr+1)) / (1.0 - beta1**(itr+1))
    for i in numba.prange(n):
        for d in numba.prange(dim):
            m[i][d] += (1 - beta1) * (grad[i][d] - m[i][d])
            v[i][d] += (1 - beta2) * (grad[i][d]**2 - v[i][d])
            Y[i][d] -= lr_t * m[i][d]/(math.sqrt(v[i][d]) + 1e-7)


@numba.njit("f4[:,:](f4[:,:],i4[:,:],i4[:,:],f4,f4,f4,f4,f4[:],f4[:,:],f4[:,:],f4[:])", parallel=True, nogil=True, cache=True)
def camel_grad(Y, pair_neighbors, pair_FP, w_neighbors, w_curv, w_FP, 
               tail_coe, edge_curvature, w_neighbors_distances, w_distances_FP, edge_curvature_X):
    '''Calculate the gradient for CAMEL embedding'''
    n, dim = Y.shape
    grad = np.zeros((n+1, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    loss = np.zeros(2, dtype=np.float32)
    number_FP=pair_FP.shape[0]/n
    number_neighbor=pair_neighbors.shape[0]/n
    
    # adjust the weight factors based on the number of data, neighbor points, and far points sampling number

    w_FP=w_FP/number_FP*number_neighbor
    w_neighbors=w_neighbors
    
    # NN grad computation
    for t in numba.prange(pair_neighbors.shape[0]):
        i = pair_neighbors[t, 0]
        j = pair_neighbors[t, 1]
        d_ij = 0.0

        for d in numba.prange(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[0] += w_neighbors *w_neighbors_distances[t,0]* 2*(tail_coe*d_ij/(1. + tail_coe*d_ij))
        +2*w_curv*min(max(edge_curvature[t]-edge_curvature_X[t],-3.0), 3.0)*d_ij
        w1 = (w_neighbors*w_neighbors_distances[t,0]) /(1. + tail_coe*d_ij)**2 + w_curv*min(max(edge_curvature[t]-edge_curvature_X[t],-3.0), 3.0)

        for d in numba.prange(dim):
            grad[i, d] += w1 * y_ij[d]
            grad[j, d] -= w1 * y_ij[d]

    # FP grad computation
    for ttt in numba.prange(pair_FP.shape[0]):
        i = pair_FP[ttt, 0]
        j = pair_FP[ttt, 1]
        d_ij = 0.0
        for d in numba.prange(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[1] += w_FP * w_distances_FP[ttt,1]* 2*d_ij/(1.0 + d_ij)
        w1 = w_FP * w_distances_FP[ttt,1]/(1. + d_ij)**2
        for d in numba.prange(dim):
            grad[i, d] -= w1 * y_ij[d]
            grad[j, d] += w1 * y_ij[d]    
        
    grad[-1, 0] = loss.sum()
    
    return grad


@numba.njit("f4[:,:](f4[:,:],i4[:,:],i4[:,:],i4,i4,f4,f4,f4,f4,f4[:],f4[:,:],f4[:,:],f4[:])", parallel=True, nogil=True, cache=True)
def camel_grad_transform(Y, pair_neighbors, pair_FP, number_neighbor, number_FP,
                   w_neighbors, w_curv, w_FP, tail_coe, edge_curvature,w_neighbors_distances, w_distances_FP, edge_curvature_X):
    '''Calculate the gradient for camel embedding for new testing data, 
    original embedding did not change.
    The way we achive this is to only change the grad for new points. 
    The first portion of Y never gets updated
    Other computing follows the same structure as in camel_grad.
    This is used both in transform and inverse_transform'''
    n_total, dim = Y.shape
    grad = np.zeros((n_total+1, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    loss = np.zeros(2, dtype=np.float32)
    # determine the number of points for new data to transform
    
    # adjust the weight factors based on the number of data , trick to reduce the repulsive load for faster convergence

    w_FP=0.01*w_FP/number_FP*number_neighbor
    w_neighbors=w_neighbors
    # NN grad
    for t in numba.prange(pair_neighbors.shape[0]):
        i = pair_neighbors[t, 0]
        j = pair_neighbors[t, 1]
        d_ij = 0.0

        for d in numba.prange(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[0] += w_neighbors *w_neighbors_distances[t,0] * 2*(tail_coe*d_ij/(1. + tail_coe*d_ij))
        +2*w_curv*max(edge_curvature[t]-edge_curvature_X[t],0.0)*d_ij
        
        ## notice that the maximum curvature is limited to 0.0 during the transform, 
        # this is because that transform is trying attract all points to the 
        # equilibrium point. The neighbor avg ditance is not updated and 
        # there are some points always have negative curvature from fit().
        # This prohibits the point gets closer. Thus, negative curvature is 
        #disabled during the transform to make the process stable.
        w1 = (w_neighbors*w_neighbors_distances[t,0]) /(1. + tail_coe*d_ij)**2 + w_curv*max(edge_curvature[t]-edge_curvature_X[t],0.0)


        for d in numba.prange(dim):
            grad[i, d] += w1 * y_ij[d]
            # assume the first node is always in the new data sets and that is why it get gradients
            # the second node is always in the basis data points and gradient for them is always zero since those points are freezed.
            # Thus, the basis points never moves 
            # grad[j, d] -= w1 * y_ij[d]


    # For FP
    for ttt in numba.prange(pair_FP.shape[0]):
        i = pair_FP[ttt, 0]
        j = pair_FP[ttt, 1]
        d_ij = 0.0
        for d in numba.prange(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[1] += w_FP * w_distances_FP[ttt,1]* 2*d_ij/(1.0 + d_ij)
        w1 = w_FP* w_distances_FP[ttt,1] /(1. + d_ij)**2
        for d in numba.prange(dim):
            grad[i, d] -= w1 * y_ij[d]
            #same reason as above, only calculating gradient for the first point
            #grad[j, d] += w1 * y_ij[d]    

    grad[-1, 0] = loss.sum()
    return grad

def preprocess_X(X, apply_pca, verbose, seed, high_dim, low_dim):
    '''Preprocess a dataset.
    '''
    tsvd = None
    pca_solution = False
    if high_dim > 100 and apply_pca:
        xmin = 0  # placeholder
        xmax = 0  # placeholder
        xmin, xmax = (np.min(X), np.max(X))
        # X -= xmin
        # X /= xmax
        xmean = np.mean(X, axis=0)
        X -= np.mean(X, axis=0)
        tsvd = TruncatedSVD(n_components=100, random_state=seed)
        X = tsvd.fit_transform(X)
        pca_solution = True
        print_verbose("Applied PCA, the dimensionality becomes 100", verbose)
    else:
        xmin, xmax = (np.min(X), np.max(X))
        X -= xmin
        X /= xmax
        xmean = np.mean(X, axis=0)
        X -= xmean
        tsvd = PCA(n_components=low_dim, random_state=seed)  # for init only
        tsvd.fit(X)
        print_verbose("X is normalized", verbose)
    return X, pca_solution, tsvd, xmin, xmax, xmean


def inverse_preprocess_X(X, verbose, pca_solution, tsvd, xmin, xmax, xmean):
    '''Inverse the preprocess of X, used to reconstruct the original feature data.
    '''
    if tsvd is None:
        raise ValueError("The tsvd object is needed to inversely tranform data back to the original space")     
        
    if pca_solution:
        X = tsvd.inverse_transform(X)       
        X += xmean        
        print_verbose("Applied Inverse PCA, the dimensionality changes from 100 to the original dimention", verbose)
    else:
        X += xmean
        X *= xmax
        X += xmin
        print_verbose("X is un-normalized", verbose)
    return X


def preprocess_X_new(X, xmin, xmax, xmean, tsvd, apply_pca, verbose):
    '''Preprocess a new dataset, given the information extracted from the basis.
    '''
    _, high_dim = X.shape
    if high_dim > 100 and apply_pca:
        # X -= xmin
        # X /= xmax        
        X -= xmean  # original xmean
        X = tsvd.transform(X)
        print_verbose(
            "Applied PCA, the dimensionality becomes 100 for new dataset.", verbose)
    else:
        X -= xmin
        X /= xmax
        X -= xmean
        print_verbose("X is normalized.", verbose)
    return X

def print_verbose(msg, verbose, **kwargs):
    if verbose:
        print(msg, **kwargs)

def construct_neighbors(X, n_neighbors):
    """
    Used to construc the neighbors using the ANNOY algorithms

    Parameters
    ----------
    X : ndarray, float32
        coordinates either in high dim or low dim
    n_neighbors : int
        number of neighbors during the  ANNOY approximation

    Returns
    -------
    nbrs : ndarray, int
        neighbor list, it should be noticed that the first column is removed 
        as it is the own node number in ANNOY 
 
    tree : ANNOY object
        For storing the neighbor searching tree

    """
    n, dim = X.shape
    n_neighbors = min(n_neighbors, n - 1)
    tree = AnnoyIndex(dim, metric='euclidean')
    if _RANDOM_STATE is not None:
        tree.set_seed(_RANDOM_STATE)
    for i in range(n):
        tree.add_item(i, X[i, :])
    tree.build(20)

    nbrs = np.zeros((n, n_neighbors), dtype=np.int32)
    nbrs_distances = np.zeros((n, n_neighbors), dtype=np.float32)

    for i in range(n):
        nbrs_, nbrs_distances_ = tree.get_nns_by_item(i, n_neighbors + 1, include_distances=True)
        nbrs[i, :] = nbrs_[1:]
        nbrs_distances[i, :] = nbrs_distances_[1:]
        
    return nbrs, tree, nbrs_distances  

def generate_pair_basis(basis,
        X,
        n_neighbors,
        n_FP,
        tree: AnnoyIndex,
        distances=None,
        verbose=True
):
    '''Generate pairs for the dataset with basis, used in transform operation
    '''
    n_old, dim_old = basis.shape
    n_new, dim_new = X.shape

    assert dim_old == dim_new, "The dimension of the original dataset is different from the new one's."
    
    n_neighbors= min(n_neighbors, n_old - 1)
    nbrs_new = np.zeros((n_new, n_neighbors), dtype=np.int32)
    nbrs_distances_new = np.zeros((n_new, n_neighbors), dtype=np.float32)    


    for i in range(n_new):
        nbrs_new[i, :], nbrs_distances_new[i,:] = tree.get_nns_by_vector(
            X[i, :], n_neighbors, include_distances=True)

    print_verbose("Found nearest neighbor for new points", verbose)
    
    pair_neighbors = sample_neighbors_pair_basis(
        n_old, X, nbrs_new, n_neighbors)
    
    if _RANDOM_STATE is None:
        pair_FP = sample_FP_pair_basis(n_old, X, pair_neighbors, n_neighbors, n_FP)
    else:
        pair_FP = sample_FP_pair_deterministic_basis(n_old,
            X, pair_neighbors, n_neighbors, n_FP, _RANDOM_STATE)
    return pair_neighbors, pair_FP, nbrs_new, nbrs_distances_new 


def generate_pair(
        X,
        nbrs,
        n_neighbors,
        n_FP,
        distances=None,
        verbose=True
):
    '''Generate pairs for the dataset.
    '''

    pair_neighbors = sample_neighbors_pair(X, nbrs, n_neighbors)

    if _RANDOM_STATE is None:
        pair_FP = sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
    else:
        pair_FP = sample_FP_pair_deterministic(
            X, pair_neighbors, n_neighbors, n_FP, _RANDOM_STATE)
    
    return pair_neighbors, pair_FP

def generate_pair_no_neighbors(
        X,
        n_neighbors,
        n_FP,
        pair_neighbors,
        distances=None,
        verbose=True
):
    '''Generate further pairs for a given dataset.
    This function is useful when the nearest neighbors comes from a given set.
    '''

    if _RANDOM_STATE is None:
        pair_FP = sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
    else:
        pair_FP = sample_FP_pair_deterministic(
            X, pair_neighbors, n_neighbors, n_FP, _RANDOM_STATE)
    return pair_neighbors, pair_FP
   
def knn_imputer(self, X, target):
    '''
    This module tries to impute missing values in labels/target using knn method
    for 'categorical', it is replace the missing value with the largest probability label
    for 'numerical', it replaces the missing values with the average values of all neighbors

    Parameters
    ----------
    X : numpy arrays, float32
    feature matrix of input data

    target : PANDAS Data Frame
    labes information with missing/empty/values

    Returns
    -------
    target : PANDA Data Frame
    labels information with imputed values at the missing location

    '''
    n_data,n_dim = X.shape
    pd.options.mode.use_inf_as_na = True
    index_imputer=target.isna()
    # modify the target_weight using the ratio of missing value to the total values
    label_ratio=1-sum(target.isna().sum())/n_data
    self.target_weight=self.target_weight*(0.5+np.arctan(100*(label_ratio-0.05))/np.pi)
          
    # X_imputer=X[index_imputer]    
    X_base=X[np.logical_not(index_imputer.loc[:,'labels'])] 
    # target_base=target[np.logical_not(index_imputer)]    
    nbrs_base, tree_base, nbrs_distances_base = construct_neighbors(X_base, self.n_neighbors)
    
    # n_imputer, dim_imputer = X_imputer.shape

    # nbrs_imputer = np.zeros((n_imputer, self.n_neighbors), dtype=np.int32)
    if self.target_type == 'categorical':
        for i in range(n_data):
            if index_imputer.loc[i,'labels']:
                nbrs_imputer, nbrs_distances_imputer = tree_base.get_nns_by_vector(
                X[i, :], self.n_neighbors, include_distances=True)
                target_nbrs_imputer=target.loc[nbrs_imputer,'labels']
                target.loc[i,'labels']=target_nbrs_imputer.value_counts().idxmax()
        

    if self.target_type == 'numerical':
        for i in range(n_data):
            if index_imputer.loc[i,'labels']:
                nbrs_imputer = tree_base.get_nns_by_vector(
                X[i, :], self.n_neighbors, include_distances=False)
                target_nbrs_imputer=target.loc[nbrs_imputer,'labels']
                target.loc[i,'labels']=target_nbrs_imputer.mean()
                

    return target


def camel(
        X,
        n_dims,
        pair_neighbors,
        pair_FP,
        tail_coe,
        w_neighbors,
        w_curv,
        w_FP,
        nbrs,
        distances,
        lr,
        num_iters,
        Yinit,
        verbose,
        hd_weight,
        intermediate,
        inter_snapshots,
        pca_solution,
        tsvd=None
      ):
    """
    Main module for the CAMEL fit operation

    Parameters
    ----------
    Please see descriotion in CAMEL class
    """
    start_time = time.time()
    n, _ = X.shape

    if intermediate:
        intermediate_states = np.empty(
            (len(inter_snapshots), n, n_dims), dtype=np.float32)
    else:
        intermediate_states = None

    # Initialize the embedding
    if isinstance(Yinit, np.ndarray):
        Yinit = Yinit.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = scaler.transform(Yinit) * 0.0001
    elif Yinit is None or (isinstance(Yinit, str) and Yinit == "pca"):
        if pca_solution:
            Y = 0.0001*X[:, :n_dims]
        else:
            Y = 0.0001*tsvd.transform(X).astype(np.float32)
    elif (isinstance(Yinit, str) and Yinit == "random"):  # random or hamming distance
        if _RANDOM_STATE is not None:
            np.random.seed(_RANDOM_STATE)
        Y = np.random.normal(size=[n, n_dims]).astype(np.float32) *0.0001
    else:
        raise ValueError((f"The argument init is of the type {type(Yinit)}. "
                          "Currently, camel only supports user supplied "
                          "numpy.ndarray object as input, or one of "
                          "['pca', 'random']."))

    # Initialize parameters for optimizer
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    if intermediate and inter_snapshots[0] == 0:
        itr_ind = 1  # move counter to one step
        intermediate_states[0, :, :] = Y

    print_verbose(
        (pair_neighbors.shape, pair_FP.shape), verbose)
    
    edge_curv_history=np.zeros(num_iters, dtype=np.float32)
    loss_history=np.zeros(num_iters, dtype=np.float32)
    
    #compute the weight factor from distances matrix and curvature matrix
    n_neighbors=nbrs.shape[1]
    w_neighbors_distances=weight_neighbors_pair(X, distances, n_neighbors)
    w_distances_FP=weight_FP_pair(X, pair_FP)
    avg_coor_X=avg_coor_compu(X, nbrs)
    edge_curvature_X=curv_nb(X, avg_coor_X, nbrs)
    
    if not hd_weight:
        w_neighbors_distances[:,:]=1.0
        w_distances_FP[:,]=1.0
        edge_curvature_X[:]=0.0

        
        
        
    for itr in range(num_iters):
        avg_coor=avg_coor_compu(Y, nbrs)
        edge_curvature=curv_nb(Y, avg_coor, nbrs)

        grad = camel_grad(Y, pair_neighbors,
                           pair_FP, w_neighbors, w_curv, w_FP, tail_coe,
                           edge_curvature, w_neighbors_distances, w_distances_FP, edge_curvature_X)
        loss_history[itr] = grad[-1, 0]
        edge_curv_history[itr]=np.mean(edge_curvature)
        if verbose and itr == 0:
            print(f"Initial Loss: {loss_history[itr]}")
        update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr)

        if intermediate:
            if (itr + 1) == inter_snapshots[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
        if (itr + 1) % 10 == 0:
            print_verbose("Iteration: %4d, Loss: %f, mean_curvature: %f," % (itr + 1, loss_history[itr], edge_curv_history[itr]), verbose)

    elapsed = time.time() - start_time
    print_verbose(f"Elapsed time: {elapsed:.2f}s", verbose)

    return Y, intermediate_states, pair_neighbors, pair_FP

def camel_transform(
        X,
        basis,
        embedding,
        n_dims,
        pair_neighbors,
        pair_FP,
        tail_coe,
        w_neighbors,
        w_curv,
        w_FP,
        nbrs,
        distances,
        lr,
        num_iters,
        Yinit,
        verbose,
        hd_weight,
        intermediate,
        inter_snapshots,
        pca_solution=False,
        tsvd=None
):
    """
    Main module for the CAMEL transform operation for new data

    Parameters
    ----------
    Please see descriotion in CAMEL class
    """
    start_time = time.time()
    n, high_dim = X.shape
    number_neighbor = pair_neighbors.shape[0]/n
    number_FP = pair_FP.shape[0]/n

    if intermediate:
        intermediate_states = np.empty(
            (len(inter_snapshots), n, n_dims), dtype=np.float32)
    else:
        intermediate_states = None

    # Initialize the embedding
    if isinstance(Yinit, np.ndarray):
        Yinit = Yinit.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = np.concatenate([embedding, scaler.transform(Yinit) * 0.0001])
    elif Yinit is None or (isinstance(Yinit, str) and Yinit == "pca"):
        if pca_solution:
            Y = np.concatenate([embedding, 0.0001 * X[:, :n_dims]])
        else:
            Y = np.concatenate(
                [embedding, 0.0001 * tsvd.transform(X).astype(np.float32)])
    elif (isinstance(Yinit, str) and Yinit == "random"):  # random or hamming distance
        if _RANDOM_STATE is not None:
            np.random.seed(_RANDOM_STATE)
        Y = np.concatenate(
            [embedding, 0.0001 * np.random.normal(size=[X.shape[0], n_dims]).astype(np.float32)])
    elif (isinstance(Yinit, str) and Yinit == "interpolate"):  # interpolate to the average point in neighbors
        #add init to the average coordinates of the basis points
        Ytemp=init_transform_embedding(embedding, nbrs)
        Y = np.concatenate([embedding, Ytemp])        
    else:
        raise ValueError((f"The argument init is of the type {type(Yinit)}. "
                          "Currently, camel only supports user supplied "
                          "numpy.ndarray object as input, or one of "
                          "['pca', 'random']."))

    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    if intermediate and inter_snapshots[0] == 0:
        itr_ind = 1  # move counter to one step
        intermediate_states[0, :, :] = Y

    print_verbose(
        (pair_neighbors.shape, pair_FP.shape), verbose)
    
    #compute the weight factor from distances matrix and curvature matrix
    n_neighbors=nbrs.shape[1]
    w_neighbors_distances=weight_neighbors_pair_basis(X, distances, n_neighbors)
    w_distances_FP=weight_FP_pair_basis(X, basis, pair_FP)
    X_total=np.concatenate([basis,X])
    avg_coor_X=avg_coor_compu(X_total, nbrs)
    edge_curvature_X=curv_nb(X_total, avg_coor_X, nbrs)
    
    if not hd_weight:
        w_neighbors_distances[:,:]=1.0
        w_distances_FP[:,]=1.0
        edge_curvature_X[:]=0.0
    
    for itr in range(num_iters):
        avg_coor=avg_coor_compu(Y, nbrs)
        edge_curvature=curv_nb(Y, avg_coor, nbrs)

    
        grad = camel_grad_transform(Y, pair_neighbors, pair_FP, number_neighbor, number_FP,
                           w_neighbors, w_curv, w_FP, tail_coe, edge_curvature, w_neighbors_distances, w_distances_FP, edge_curvature_X)
        C = grad[-1, 0]
        if verbose and itr == 0:
            print(f"Initial Loss: {C}")
        update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr)

        if intermediate:
            if (itr+1) == inter_snapshots[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
                if itr_ind > 12:
                    itr_ind -= 1
        if (itr + 1) % 10 == 0:
            print_verbose("Iteration: %4d, Loss: %f" % (itr + 1, C), verbose)

    elapsed = str(datetime.timedelta(seconds=time.time() - start_time))
    print_verbose("Elapsed time: %s" % (elapsed), verbose)
    
        
    return Y, intermediate_states

def camel_inverse_transform(
        X,
        basis,
        embedding,
        n_dims,
        pair_neighbors,
        pair_FP,
        tail_coe,
        w_neighbors,
        w_curv,
        w_FP,
        nbrs,
        distances,
        lr,
        num_iters,
        Yinit,
        verbose,
        hd_weight,
        intermediate,
        inter_snapshots
):
    """
    Main module for the CAMEL inverse_transform operation
    totice that X is acturally the low-dim embedding from fit;
    embedding is the original basis data in high dimentional.
    Thus, this modudles is trying to inverse the embedding process and is from low-dim to high dim

    Parameters
    ----------
    Please see descriotion in CAMEL class
    """
    start_time = time.time()
    n, low_dim = X.shape
    number_neighbor = pair_neighbors.shape[0]/n
    number_FP = pair_FP.shape[0]/n

    if intermediate:
        intermediate_states = np.empty(
            (len(inter_snapshots), n, n_dims), dtype=np.float32)
    else:
        intermediate_states = None

    # Initialize the embedding
    if isinstance(Yinit, np.ndarray):
        Yinit = Yinit.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = np.concatenate([embedding, scaler.transform(Yinit) * 0.0001])
    elif Yinit is None or (isinstance(Yinit, str) and Yinit == "interpolate"):
        #add init to the average coordinates of the basis points
        Ytemp=init_transform_embedding(embedding, nbrs)
        Y = np.concatenate([embedding, Ytemp])   
    elif (isinstance(Yinit, str) and Yinit == "random"):  # random or hamming distance
        if _RANDOM_STATE is not None:
            np.random.seed(_RANDOM_STATE)
        Y = np.concatenate(
            [embedding, 0.00001 * np.random.normal(size=[X.shape[0], n_dims]).astype(np.float32)])
    else:
        raise ValueError((f"The argument init is of the type {type(Yinit)}. "
                          "Currently, camel only supports user supplied "
                          "numpy.ndarray object as input, or one of "
                          "['pca', 'random']."))

    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    if intermediate and inter_snapshots[0] == 0:
        itr_ind = 1  # move counter to one step
        intermediate_states[0, :, :] = Y

    print_verbose(
        (pair_neighbors.shape, pair_FP.shape), verbose)

    #compute the weight factor from distances matrix and curvature matrix
    n_neighbors=nbrs.shape[1]
    w_neighbors_distances=weight_neighbors_pair(X, distances, n_neighbors)
    w_distances_FP=weight_FP_pair_basis(X, basis, pair_FP)
    X_total=np.concatenate([basis,X])
    avg_coor_X=avg_coor_compu(X_total, nbrs)
    edge_curvature_X=curv_nb(X_total, avg_coor_X, nbrs)
    
    if not hd_weight:
        w_neighbors_distances[:,:]=1.0
        w_distances_FP[:,]=1.0
        edge_curvature_X[:]=0.0
    
    for itr in range(num_iters):
        avg_coor=avg_coor_compu(Y, nbrs)
        edge_curvature=curv_nb(Y, avg_coor, nbrs)

        
        grad = camel_grad_transform(Y, pair_neighbors, pair_FP, number_neighbor, number_FP,
                           w_neighbors, w_curv, w_FP, tail_coe, edge_curvature, w_neighbors_distances, w_distances_FP,edge_curvature_X)
        C = grad[-1, 0]
        if verbose and itr == 0:
            print(f"Initial Loss: {C}")
        update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr)

        if intermediate:
            if (itr+1) == inter_snapshots[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
                if itr_ind > 12:
                    itr_ind -= 1
        if (itr + 1) % 10 == 0:
            print_verbose("Iteration: %4d, Loss: %f" % (itr + 1, C), verbose)

    elapsed = str(datetime.timedelta(seconds=time.time() - start_time))
    print_verbose("Elapsed time: %s" % (elapsed), verbose)
    
        
    return Y, intermediate_states

class CAMEL(BaseEstimator):
    '''Curvature Augmented Manifold Embedding and Learning (CAMEL).
    Performs forward and inverse projection of large data sets. The current capability includes
    1. Unsupervised learning / forward projection of high dimentional data to 
    low dimentional space, which is achived using fit(X) and fit_transform(X) method
    2. Unsupervised transformation / forward projection of high dimentional data to 
    low dimentional space, which is achived using transform(X) method
    3. Supervised learning/semi supervised learning/metric learning of high dimentional data to
    low dimentional space, which is achvied using fit(X,y)/fit_transform(X, y)/transform(X) method
    4. Inverse embedding/generative model from low dimentioanl space to high dimentional space, 
    which is achvied using inverse_transform method. 
    
    Detailed theretical discussion can be found in the axriv pre-print and more examples and benchmarking 
    can be found in my github link
    

    Parameters
    ---------
    n_components: int, default=2
        Dimensions of the embedded space. We recommend to use 2 or 3.

    n_neighbors: int, default=10
        Number of neighbors considered for nearest neighbor pairs for local structure preservation.

    FP_number: float, default=10
        Number of further points(e.g. 10 Further pairs per node)
        Further pairs are used for both local and global structure preservation.

    pair_neighbors: numpy.ndarray, optional
        Nearest neighbor pairs constructed from a previous run or from outside functions.

    pair_FP: numpy.ndarray, optional
        Further pairs constructed from a previous run or from outside functions.

    nbrs: numpy.ndaarray, init, default None
        neighbor list, row number is the node number and column is the neasrest node number    
    
    tree: ANNOY object, default None
        tres structure during the ANNOY search and is used to obtain neighbor list, distance and other information
    
    basis: numpy.ndarray, default None
        orginal feature space and is used in the inverse_transform as the high dim embedding
        
    distances: float32
        Neighbor distances matrix. of the original X matrix

    tail_coe: float, default=0.05
        Parameter to control the attractive force of neighbors (1/(1+tail_coe*dij)**2), smaller values indicate flat tail
    
    w_neighbors: float, default=0.1
        weight coefficient for attractive force of neighbors, large values indicates strong force for the same distance metric
        
    w_curv: float, default=0.005
        weight coefficient for attractive/repulsive force due to local curvature, large values indicates strong force for the same distance metric        

    w_FP: float, default=2
        weight coefficient for repulsive force of far points, large values indicates strong force for the same distance metric    
    
    lr: float, default=1.0
        Learning rate of the Adam optimizer for embedding.

    num_iters: int, default=400
        Number of iterations for the optimization of embedding. I observe that 200 is sufficient for most cases and 400 is used here for safe reason.

    target_encoder: string, default='OneHotEncoder'
        Method in sklearn to do the target encoding, used for supervised learning and metric learning (and transform)
        Other options, 'OrdinalEncoder', TargetEncoder'  - not used in the current version, can be easily added
        
    target_weight: float, default=0.5
        weight factor for target/label during the supervised learning, 0 indicates no weight and it reduces to unsupervised one,
        1 indicates infinity weight (set as a large value in practice
                                     
    target_method: string, default=normal
    method used for supervised learning, 'normal' indicates that label information is only used to update the knn graph, but the embedding only uses the original X features
    'extra' is the option to use label information for both knn graph and feature spaces for X.
    
    verbose: bool, default=False
        Whether to print additional information during initialization and fitting.

    hd_wieht: bool, default=True
        whether to include high dimension distance and curvature to weight the force field computing in the low dimension
    
    
    apply_pca: bool, default=True
        Whether to apply PCA on the data before pair construction.

    intermediate: bool, default=False
        Whether to return intermediate state of the embedding during optimization.
        If True, returns a series of embedding during different stages of optimization.

    intermediate_snapshots: list[int], optional
        The index of step where an intermediate snapshot of the embedding is taken.
        If intermediate sets to True, the default value will be [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450]

    random_state: int, optional
        Random state for the camel instance.
        Setting random state is useful for repeatability.


    '''

    def __init__(self,
                 n_components=2,
                 n_neighbors=10,
                 FP_number=20,
                 pair_neighbors=None,
                 pair_FP=None,
                 nbrs=None,
                 tree=None,
                 basis=None,
                 distances=None,
                 tail_coe=0.05,
                 w_neighbors=1.0,
                 w_curv=0.001,
                 w_FP=20,
                 lr=1.0,
                 num_iters=400,
                 target_type='categorical',
                 target_encoder='OneHotEncoder',
                 target_method = 'normal',
                 target_weight=0.5,
                 verbose=False,
                 apply_pca=True,
                 hd_weight=True,
                 intermediate=False,
                 intermediate_snapshots=[
                     0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 400],
                 random_state=None
                 ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.FP_number = FP_number
        self.pair_neighbors = pair_neighbors
        self.pair_FP = pair_FP
        self.nbrs = nbrs
        self.tree = tree
        self.basis = basis
        self.distances = distances
        self.tail_coe=tail_coe
        self.w_neighbors=w_neighbors
        self.w_curv=w_curv
        self.w_FP=w_FP
        self.lr = lr
        self.num_iters = num_iters
        self.target_type=target_type
        self.target_encoder=target_encoder
        self.target_weight=target_weight
        self.target_method = target_method
        self.apply_pca = apply_pca
        self.verbose = verbose
        self.hd_weight = hd_weight
        self.intermediate = intermediate
        self.intermediate_snapshots = intermediate_snapshots

        global _RANDOM_STATE
        if random_state is not None:
            assert(isinstance(random_state, int))
            self.random_state = random_state
            _RANDOM_STATE = random_state  # Set random state for numba functions
            warnings.warn(f'Warning: random state is set to {_RANDOM_STATE}')
        else:
            try:
                if _RANDOM_STATE is not None:
                    warnings.warn('Warning: random state is removed')
            except NameError:
                pass
            self.random_state = 0
            _RANDOM_STATE = None  # Reset random state

        if self.n_components < 2:
            raise ValueError(
                "The number of projection dimensions must be at least 2.")
        if self.lr <= 0:
            raise ValueError("The learning rate must be larger than 0.")
        if not self.apply_pca:
            warnings.warn(
                "Running ANNOY Indexing on high-dimensional data. Nearest-neighbor search may be slow!")

    def decide_num_pairs(self, n):
        if self.n_neighbors is None:
            if n <= 10000:
                self.n_neighbors = 10
            else:
                self.n_neighbors = int(round(10 + 15 * (np.log10(n) - 4)))
        self.n_FP = int(round(self.FP_number))
        if self.n_neighbors < 1:
            raise ValueError(
                "The number of nearest neighbors can't be less than 1")
        if self.n_FP < 1:
            raise ValueError(
                "The number of further points can't be less than 1")

    def fit(self, X, target=None, init=None):
        '''Projects a high dimensional dataset into a low-dimensional embedding, 
        without returning the output.

        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected. 
            An embedding will get created based on parameters of the camel instance.
            
        target: numpy.ndarray or categorical data. This is the target/label used for supervised learning
            Default is not provided for unsupervised learning. If provided, encoding is done using the target_encoder and target_weight factors
            to ccatenate this target info into X. Then, regular operation can be performed.

        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset.
            If 'random', then the low dimensional embedding is initialized with a 
            Gaussian distribution.

        '''
        self.basis=np.copy(X).astype(np.float32)
        X = np.copy(X).astype(np.float32)
        
        if target is not None and isinstance (target, (np.ndarray)):
            target=pd.DataFrame(data=target,columns=['labels'])

        # Perform feature preprocessing and label encoding if labels are provided.
        X, self.nbrs, self.tree = self.X_target_processing(X, target)
        
        
        print_verbose(
            "camel(n_neighbors={}, n_FP={}, distances={}, "
            "tail_coe={}, w_neighbors={},w_curv={}, w_FP={}"
            "lr={}, n_iters={}, apply_pca={}, opt_method='adam', "
            "verbose={}, intermediate={}, seed={})".format(
                self.n_neighbors,
                self.n_FP,
                self.distances,
                self.tail_coe,
                self.w_neighbors,
                self.w_curv,
                self.w_FP,
                self.lr,
                self.num_iters,
                self.apply_pca,
                self.verbose,
                self.intermediate,
                _RANDOM_STATE
            ), self.verbose
        )


        # Sample pairs

        self.sample_pairs(X)
        self.num_instances = X.shape[0]
        self.num_dimensions = X.shape[1]
        # Initialize and Optimize the embedding
        self.embedding_, self.intermediate_states, self.pair_neighbors, self.pair_FP = camel(
            X,
            self.n_components,
            self.pair_neighbors,
            self.pair_FP,
            self.tail_coe,
            self.w_neighbors,
            self.w_curv,
            self.w_FP,            
            self.nbrs,
            self.distances,
            self.lr,
            self.num_iters,
            init,
            self.verbose,
            self.hd_weight,
            self.intermediate,
            self.intermediate_snapshots,
            self.pca_solution,
            self.tsvd_transformer
        )
        

        
        return self

    def fit_transform(self, X, target=None, init=None):
        '''Projects a high dimensional dataset into a low-dimensional embedding 
        and return the embedding.

        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected. 
            An embedding will get created based on parameters of the camel instance.

        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset.
            If 'random', then the low dimensional embedding is initialized with 
            a Gaussian distribution.

        
        '''

        self.fit(X, target, init)
        if self.intermediate:
            return self.intermediate_states
        else:
            return self.embedding_

    def inverse_transform(self, X, basis=None, embedding=None, init=None):
        '''Projects a low dimensional dataset into high dimentional space 
        and return the embedding.

        Parameters
        ---------
        X: numpy.ndarray
            The low dimentional representation that is being projected back to 
            the original feature space 
            An embedding will get created based on parameters of the camel instance.

        basis: numpy.ndarray
            The original dataset that have already been applied during the `fit` 
            or `fit_transform` process.

        init: str, optional
            One of ['interpolate', 'random']. Initialization of the embedding at 
            high dimentional, default='interpolate'.
            If 'interpolate', the low dim knn is used to find the corresponding 
            high dim points, average values are used initial guess
            If 'random', then the low dimensional embedding is initialized with 
            a uniform distribution.

 
        '''

        if X is None:
            raise ValueError("Low dim representation is required for inverse_transform")      
        X = np.copy(X).astype(np.float32)
        n_inverse_sample, dim_inverse_sample = X.shape
        
        if n_inverse_sample <= 0:
            raise ValueError("The sample size for inverse operation must be larger than 0")
        
        if basis is None and embedding is None:
            basis=self.basis
            embedding=self.embedding_
            basis = np.copy(basis).astype(np.float32)
            embedding = np.copy(embedding).astype(np.float32)
            n_basis, dim_basis = basis.shape
            n_embedding, dim_embedding = embedding.shape
        elif basis is not None and embedding is not None:
            basis = np.copy(basis).astype(np.float32)
            embedding = np.copy(embedding).astype(np.float32)
            n_basis, dim_basis = basis.shape
            n_embedding, dim_embedding = embedding.shape
        else:
            raise ValueError("The inverse transform requires both original basis data and embedding data")
            
        if n_embedding != n_basis:
            raise ValueError("The inverse transform requires the number of rows matches for original basis data and embedding data")            

        if dim_embedding != dim_inverse_sample:
            raise ValueError("The inverse transform requires the dim of representation matches for the known embedding dim")     
            
        if n_basis <= 0:
            raise ValueError("The original basis data size must be larger than 0")
            

        #preprocess basis to determin the pca solution for future inverse pca calculation
        if n_basis <= 0:
            raise ValueError("The sample size must be larger than 0")
           
        basis, pca_solution, tsvd, self.xmin, self.xmax, self.xmean = preprocess_X(
            basis, self.apply_pca, self.verbose, self.random_state, dim_basis, self.n_components)
        self.tsvd_transformer = tsvd
        self.pca_solution = pca_solution
        n_basis, dim_basis = basis.shape







        #build knn graph for low dim representation for inverse_transform, this knn is assumed to hold true for original high dim space
        nbrs_embedding, tree_embedding, nbrs_distances_embedding = construct_neighbors(embedding, self.n_neighbors)      
        
        # Sample pairs
        # Sample pairs for new low-dim data points to be inverse_transformed -- neighbr pair list from the original embedding points
        self.pair_neighbors, self.pair_FP,  nbrs_new, nbrs_distances_new  = generate_pair_basis(embedding,
                X, self.n_neighbors, self.n_FP, tree_embedding, self.distances, self.verbose)
                    
        #stack old nbrs and new nbrs to form a total knn list
        nbrs_total=np.row_stack((nbrs_embedding, nbrs_new))
        nbrs_distances_total=np.row_stack((nbrs_distances_embedding, nbrs_distances_new))
        
        n_components=dim_basis
        
        # Initialize and Optimize the embedding
        Y, intermediate_states = camel_inverse_transform(X, embedding, basis, n_components, self.pair_neighbors, self.pair_neighbors, self.tail_coe,
                    self.w_neighbors, self.w_curv, self.w_FP, nbrs_total, nbrs_distances_total, self.lr,
                                            self.num_iters, init, self.verbose, self.hd_weight,
                                            self.intermediate, self.intermediate_snapshots)
        
        #check the first/cloest neighbors point, if the distance is too small and can be considered to be the same point
        # Thus, the same embedding in the original fit should be kept.        
        nbrs=nbrs_new
        n_old=basis.shape[0]
        n_new=X.shape[0]
        
        for i in range(n_new): 
            distance_old_new=euclid_dist(X[i,:],embedding[nbrs[i,0],:])
            if distance_old_new < 1e-3:  #a small number to check if the new point is sufficiently close to an old point in fit
                Y[n_old+i,:]=basis[nbrs[i,0],:]
        
        #postprocess results from inverse_transform to the original feature space
        Y = inverse_preprocess_X(Y, self.verbose, self.pca_solution, 
                                  self.tsvd_transformer, self.xmin, self.xmax, self.xmean)
            
 

        #change the self embeedding memory to the new Y emebdding, the first n_old is from fit() of basis, remaining n_new is from new data, only reported in transform()
        # for inverse_transform, the self.embedding_ will be used.
        
       
        if self.intermediate:
            return intermediate_states
        else:
            return Y[n_old:, :]
            #return Y[:self.embedding_.shape[0], :]
            
    def transform(self, X, basis=None, init=None):
        '''Projects a low dimentional representation to  the original high dimensional feature space, 
        using existing embedding space and corresponding feature vector of the low dim representation.


        Parameters
        ---------
        X: numpy.ndarray
            The new high-dimensional dataset that is being projected. 
            An embedding will get created based on parameters of the camel instance.

        basis: numpy.ndarray
            The original dataset that have already been applied during the `fit` 
            or `fit_transform` process.

        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset. 
            The PCA instance will be the same one that was applied to the original dataset during the `fit` or `fit_transform` process. 
            If 'random', then the low dimensional embedding is initialized with a Gaussian distribution.

 
        '''

        if X is None:
            raise ValueError("Both new and old data are required for transform")    
            
        if basis is None:
            basis=self.basis            
            
        basis = np.copy(basis).astype(np.float32)
        # Preprocess the dataset, it should be noted that the original data (basis) needs to be reprocessed as the fit method may have larger dim depending on the 
        # way of possible supervised learning
        
        n, dim = basis.shape
        if n <= 0:
            raise ValueError("The sample size must be larger than 0")
           
        if X.shape[1] != dim:
            raise ValueError("The feature dimention of new data X and original data basis must match")       
            
        basis, pca_solution, tsvd, self.xmin, self.xmax, self.xmean = preprocess_X(
            basis, self.apply_pca, self.verbose, self.random_state, dim, self.n_components)
        self.tsvd_transformer = tsvd
        self.pca_solution = pca_solution
        # Deciding the number of pairs
        self.decide_num_pairs(n)
            
        # Preprocess the data
        X = np.copy(X).astype(np.float32)
   
        #X = np.unique(X, axis=0)
        X = preprocess_X_new(X, self.xmin, self.xmax,
                             self.xmean, self.tsvd_transformer,
                             self.apply_pca, self.verbose)

            
        #build knn graph for original data for transform, since it is not known how fit knn is built, redo knn here to avoid mismatch
        # this is on for the basis knn information, which is used to check every new points knn info
        nbrs_basis, tree_basis, nbrs_distances_basis = construct_neighbors(basis, self.n_neighbors)
    
        # Sample pairs for new data points to be transformed -- neighbr pair list from the basis points
        
        # Sample pairs
        self.pair_neighbors, self.pair_FP,  nbrs_new, nbrs_distances_new = generate_pair_basis(basis,
                X, self.n_neighbors, self.n_FP, tree_basis, self.distances, self.verbose)
                    
        #stack old nbrs and new nbrs to form a total knn list
        nbrs_total=np.row_stack((nbrs_basis, nbrs_new))
        nbrs_distances_total= np.row_stack((nbrs_distances_basis, nbrs_distances_new), dtype=np.float32)
        
        # Initialize and Optimize the embedding
        Y, intermediate_states = camel_transform(X, basis, self.embedding_, self.n_components, self.pair_neighbors, self.pair_neighbors, self.tail_coe,
                    self.w_neighbors, self.w_curv, self.w_FP, nbrs_total, nbrs_distances_total, self.lr,
                                            self.num_iters, init, self.verbose, self.hd_weight,
                                            self.intermediate, self.intermediate_snapshots,
                                            self.pca_solution, self.tsvd_transformer)
        

        #check the first/cloest neighbors point, if the distance is too small and can be considered to be the same point
        # Thus, the same embedding in the original fit should be kept.        

        embedding=self.embedding_
        n_old=embedding.shape[0]
        n_new=X.shape[0]
        
        for i in range(n_new): 
            distance_old_new=euclid_dist(X[i,:],basis[nbrs_new[i,0],:])
            if distance_old_new < 1e-3:  #a small number to check if the new point is sufficiently close to an old point in fit
                Y[n_old+i,:]=embedding[nbrs_new[i,0],:]
        #change the self embeedding memory to the new Y emebdding, the first n_old is from fit() of basis, remaining n_new is from new data, only reported in transform()
        # for inverse_transform, the self.embedding_ will be used.
        
        # self.embedding_=Y
        # self.basis=X
        
        if self.intermediate:
            return intermediate_states
        else:
            return Y[n_old:, :]
            #return Y[:self.embedding_.shape[0], :]
                        
    def X_target_processing(self, X, target):
        
        """
        Processing the data of input feature: X and possible label data: target
        Perform weight scaling between X distance and target distance
        Compute knn graph with X only or possible target information
        
        X: input feature arrays
        target: label information (can be None)
        nbrs: output neighbor list
        tree: output ANNOY tree
        X: output features, no dim change; 
        
        Perform encoding process to convert the data for future processing
        1. target_type: categorial and numerical
        2. target_encoder: no encoding for numerical values 
                           for categorical, is onehotencoding - others can be added
        3. target_method: 'normal' is to use label construct knn graph only without 
        modifying input X feature array; now only has one method, future other method can be added.
        """

        '''
        check if th target has missing/empty/NaN values, if so, it belongs to the semi-supervised learning
        have two ways to handle it, "imputation" tris to impuetes the missing values from neighbors,
        "sequential" tries to fit the supervised learning using complete labels
        '''
        if (target is not None) and (sum(target.isnull().sum())>0):
            target=knn_imputer(self, X, target)


        # Perform label encoding if labels are provided.
        if (target is not None) and (self.target_method == 'normal') and (self.target_type == 'categorical'):  #categorical values and no need to encode
           if self.target_encoder == 'OneHotEncoder':
               enc = pd.get_dummies(target)
               target=enc.to_numpy(dtype=np.float32)  #generate np array encoding for labels
               ##need to weight and scale target values
               number_samples = 100
           if target.ndim == 2:
               target_scale=target_scale_compute_2D(X, target, number_samples)
           if  target.ndim == 1:
               target_scale=target_scale_compute_1D(X, target, number_samples)
           
           target=target*target_scale*self.target_weight/(max((1-self.target_weight),1e-6))
           X_add = np.column_stack((X,target))
           # Preprocess the dataset
           n, dim = X.shape
           if n <= 0:
               raise ValueError("The sample size must be larger than 0")
               
           X_add, pca_solution_add, tsvd_add, xmin_add, xmax_add, xmean_add = preprocess_X(
               X_add, self.apply_pca, self.verbose, self.random_state, dim, self.n_components)

           X, pca_solution, tsvd, self.xmin, self.xmax, self.xmean = preprocess_X(
               X, self.apply_pca, self.verbose, self.random_state, dim, self.n_components)
           self.tsvd_transformer = tsvd
           self.pca_solution = pca_solution
           # Deciding the number of pairs
           self.decide_num_pairs(n)
           #compute the knn graph using X
           self.nbrs, self.tree, self.distances = construct_neighbors(X_add, self.n_neighbors)

        elif (target is not None) and (self.target_method == 'normal') and (self.target_type == 'numerical'): # numerical values of target for regression analysis
            ##need to weight and scale target values
            target = np.copy(target).astype(np.float32)
            number_samples = 100
            if target.ndim == 2:
                target_scale = target_scale_compute_2D(X, target, number_samples)
            if  target.ndim ==1:
                target_scale=target_scale_compute_1D(X, target, number_samples)
                    
            target=target*target_scale*self.target_weight/(max((1-self.target_weight),1e-6))
            X_add = np.column_stack((X,target))
            # Preprocess the dataset
            n, dim = X.shape
            if n <= 0:
                raise ValueError("The sample size must be larger than 0")
                
            X_add, pca_solution_add, tsvd_add, xmin_add, xmax_add, xmean_add = preprocess_X(
                X_add, self.apply_pca, self.verbose, self.random_state, dim, self.n_components)

            X, pca_solution, tsvd, self.xmin, self.xmax, self.xmean = preprocess_X(
                X, self.apply_pca, self.verbose, self.random_state, dim, self.n_components)
            self.tsvd_transformer = tsvd
            self.pca_solution = pca_solution
            # Deciding the number of pairs
            self.decide_num_pairs(n)
            #compute the knn graph using X
            self.nbrs, self.tree, self.distances = construct_neighbors(X_add, self.n_neighbors)
            
        
        elif (target is None):

            # Preprocess the dataset
            n, dim = X.shape
            if n <= 0:
                raise ValueError("The sample size must be larger than 0")

            X, pca_solution, tsvd, self.xmin, self.xmax, self.xmean = preprocess_X(
                X, self.apply_pca, self.verbose, self.random_state, dim, self.n_components)
            self.tsvd_transformer = tsvd
            self.pca_solution = pca_solution
            # Deciding the number of pairs
            self.decide_num_pairs(n)
            #compute the knn graph using X
            self.nbrs, self.tree, self.distances = construct_neighbors(X, self.n_neighbors)
 

        else:
            raise ValueError("The other supervised learning method is not supported at this stage")
        
        return X, self.nbrs, self.tree
    
    def sample_pairs(self, X):
        '''
        Sample camel pairs from the dataset.

        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected.
            
        nbrs: list of neighbor points from ANNOY algorithms

        '''
        # Creating pairs
        print_verbose("Finding pairs", self.verbose)
        if self.pair_neighbors is None:
            self.pair_neighbors, self.pair_FP = generate_pair(
                X, self.nbrs, self.n_neighbors, self.n_FP, self.distances, self.verbose
            )
            print_verbose("Pairs sampled successfully.", self.verbose)
        elif self.pair_FP is None:
            print_verbose(
                "Using user provided nearest neighbor pairs.", self.verbose)
            assert self.pair_neighbors.shape == (
                X.shape[0] * self.n_neighbors, 2), "The shape of the user provided nearest neighbor pairs is incorrect."
            self.pair_neighbors, self.pair_FP = generate_pair_no_neighbors(
                X, self.n_neighbors, self.n_FP, self.pair_neighbors, self.distances, self.verbose
            )
            print_verbose("Pairs sampled successfully.", self.verbose)
        else:
            print_verbose("Using stored pairs.", self.verbose)


        return self