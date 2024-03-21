import os
import numpy as np
import scipy.stats
from scipy.spatial.kdtree import distance_matrix
from sklearn.cluster import OPTICS
import coranking
from coranking.metrics import trustworthiness, continuity, LCMC

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from collections import Counter
from numpy.random import default_rng
from annoy import AnnoyIndex
from sklearn.decomposition import TruncatedSVD, PCA
from gap_statistic import OptimalK
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from joblib import Parallel, delayed
import numba




global _RANDOM_STATE, dim_threshold
_RANDOM_STATE = None
dim_threshold=500


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

def compute_ranking_matrix_parallel(D):
    """ Compute ranking matrix in parallel. Input (D) is distance matrix
    """
    # if data is small, no need for parallel
    if len(D) > 1000:
        n_jobs = -1
    else:
        n_jobs = 1
    r1 = Parallel(n_jobs, prefer="threads")(
            delayed(np.argsort)(i)
            for i in D.T
        )
    r2 = Parallel(n_jobs, prefer="threads")(
            delayed(np.argsort)(i)
            for i in r1
        )
    # write as a single array
    r2_array = np.zeros((len(r2), len(r2[0])), dtype='int32')
    for i, r2row in enumerate(r2):
        r2_array[i] = r2row
    return r2_array


@numba.njit(fastmath=True)
def populate_Q(Q, i, m, R1, R2):
    """ populate coranking matrix using numba for speed
    """
    for j in range(m):
        k = R1[i, j]
        l = R2[i, j]
        Q[k, l] += 1
    return Q


def iterate_compute_distances(data):
    """ Compute pairwise distance matrix iteratively, so we can see progress
    """
    n = len(data)
    D = np.zeros((n, n), dtype='float32')
    col = 0

    for i, distances in enumerate(pairwise_distances_chunked(data, n_jobs=-1)):
        D[col : col + len(distances)] = distances
        col += len(distances)

    return D

def compute_coranking_matrix(data_ld, data_hd = None, D_hd = None):
    """ Compute the full coranking matrix
    """
   
    # compute pairwise probabilities
    if D_hd is None:
        D_hd = iterate_compute_distances(data_hd)
    
    D_ld =iterate_compute_distances(data_ld)
    n = len(D_ld)
    # compute the ranking matrix for high and low D
    rm_hd = compute_ranking_matrix_parallel(D_hd)
    rm_ld = compute_ranking_matrix_parallel(D_ld)

    # compute coranking matrix from_ranking matrix
    m = len(rm_hd)
    Q = np.zeros(rm_hd.shape, dtype='int16')
    for i in range(m):
        Q = populate_Q(Q,i, m, rm_hd, rm_ld)
        
    Q = Q[1:,1:]
    return Q

@numba.njit(fastmath=True)
def qnx_crm(crm, k):
    """ Average Normalized Agreement Between K-ary Neighborhoods (QNX)
    # QNX measures the degree to which an embedding preserves the local
    # neighborhood around each observation. For a value of K, the K closest
    # neighbors of each observation are retrieved in the input and output space.
    # For each observation, the number of shared neighbors can vary between 0
    # and K. QNX is simply the average value of the number of shared neighbors,
    # normalized by K, so that if the neighborhoods are perfectly preserved, QNX
    # is 1, and if there is no neighborhood preservation, QNX is 0.
    #
    # For a random embedding, the expected value of QNX is approximately
    # K / (N - 1) where N is the number of observations. Using RNX
    # (\code{rnx_crm}) removes this dependency on K and the number of
    # observations.
    #
    # @param crm Co-ranking matrix. Create from a pair of distance matrices with
    # \code{coranking_matrix}.
    # @param k Neighborhood size.
    # @return QNX for \code{k}.
    # @references
    # Lee, J. A., & Verleysen, M. (2009).
    # Quality assessment of dimensionality reduction: Rank-based criteria.
    # \emph{Neurocomputing}, \emph{72(7)}, 1431-1443.

    Python reimplmentation of code by jlmelville
    (https://github.com/jlmelville/quadra/blob/master/R/neighbor.R)
    """
    qnx_crm_sum = np.sum(crm[:k, :k])
    return qnx_crm_sum / (k * len(crm))

@numba.njit(fastmath=True)
def rnx_crm(crm, k):
    """ Rescaled Agreement Between K-ary Neighborhoods (RNX)
    # RNX is a scaled version of QNX which measures the agreement between two
    # embeddings in terms of the shared number of k-nearest neighbors for each
    # observation. RNX gives a value of 1 if the neighbors are all preserved
    # perfectly and a value of 0 for a random embedding.
    #
    # @param crm Co-ranking matrix. Create from a pair of distance matrices with
    # \code{coranking_matrix}.
    # @param k Neighborhood size.
    # @return RNX for \code{k}.
    # @references
    # Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
    # Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
    # dimensionality reduction based on similarity preservation.
    # \emph{Neurocomputing}, \emph{112}, 92-108.

    Python reimplmentation of code by jlmelville
    (https://github.com/jlmelville/quadra/blob/master/R/neighbor.R)
    """
    n = len(crm)
    return ((qnx_crm(crm, k) * (n - 1)) - k) / (n - 1 - k)


#@numba.njit(fastmath=True)
def rnx_auc_crm(crm):
    """ Area Under the RNX Curve 
    # The RNX curve is formed by calculating the \code{rnx_crm} metric for
    # different sizes of neighborhood. Each value of RNX is scaled according to
    # the natural log of the neighborhood size, to give a higher weight to smaller
    # neighborhoods. An AUC of 1 indicates perfect neighborhood preservation, an
    # AUC of 0 is due to random results.
    #
    # param crm Co-ranking matrix.
    # return Area under the curve.
    # references
    # Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
    # Multi-scale similarities in stochastic neighbour embedding: Reducing
    # dimensionality while preserving both local and global structure.
    # \emph{Neurocomputing}, \emph{169}, 246-261.

    Python reimplmentation of code by jlmelville
    (https://github.com/jlmelville/quadra/blob/master/R/neighbor.R)
    """
    n = len(crm)
    num = 0
    den = 0
    
    qnx_crm_sum = 0
    for k in range(1, n - 2):
        #for k in (range(1, n - 2)):
        qnx_crm_sum += np.sum(crm[(k-1), :k]) + np.sum(crm[:k, (k-1)]) - crm[(k-1), (k-1)]
        qnx_crm = qnx_crm_sum / (k * len(crm))
        rnx_crm = ((qnx_crm * (n - 1)) - k) / (n - 1 - k)
        num += rnx_crm / k
        den += 1 / k
    return num / den




def preprocess_X(X, seed=_RANDOM_STATE):
    '''Preprocess a dataset.
    '''
    tsvd = None
    X = X - np.mean(X, axis=0)
    tsvd = TruncatedSVD(n_components=dim_threshold, random_state=seed)
    X = tsvd.fit_transform(X)

    return X

    
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
    distance='euclidean'
    tree = AnnoyIndex(dim, metric=distance)
    if _RANDOM_STATE is not None:
        tree.set_seed(_RANDOM_STATE)
    for i in range(n):
        tree.add_item(i, X[i, :])
    tree.build(20)

    nbrs = np.zeros((n, n_neighbors), dtype=np.int32)

    for i in range(n):
        nbrs_ = tree.get_nns_by_item(i, n_neighbors + 1)
        nbrs[i, :] = nbrs_[1:]
        
    return nbrs    

def knn_eval_large(X, y, n_neighbors=5, n_splits=10, seed=0, sample_size=10000):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by a k-nearest neighbor classifier, defined over
    a small sample of datasets.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        n_neighbors: Number of neighbors considered by the classifier.
        n_splits: Number of splits used in the cross validation.
    Output:
        acc: The avg accuracy generated by the clf, using cross val.
    '''
    X.astype(np.float32)   
    if X.shape[1]>dim_threshold:
        X=preprocess_X(X)
        
    rng = np.random.default_rng(seed=seed)
    skf = StratifiedKFold(n_splits=n_splits)
    correct_cnt = 0 # Counter for intersection
    cnt = 0
    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        sample_size = min(X_test.shape[0], sample_size) # prevent overflow
        indices = rng.choice(np.arange(X_test.shape[0]), size=sample_size, replace=False)
        for i in indices:
            index_list_neighbors = calculate_neighbors(X_train, X_test[i], n_neighbors - 1) # no self in train set
            y_neighbor = y_train[index_list_neighbors]
            # find the predicted value
            # if there's a tie, pick one of the pred randomly
            y_cnt = np.bincount(y_neighbor)
            mode = np.amax(y_cnt)
            y_pred = np.arange(y_cnt.shape[0])[y_cnt == mode]
            y_pred = rng.choice(y_pred)
            # compare the predicted with the label
            if y_pred == y_test[i]:
                correct_cnt += 1
            cnt += 1

    return correct_cnt / cnt

def svm_eval_large(X, y, n_splits=10, sample_size=100000, 
                   seed=20200202, **kwargs):
    '''
    This is an accelerated version of the SVM function.
    Training an SVM is infeasible over huge dataset. We therefore only sample
    a portion of data to perform the training and testing.
    '''
    X.astype(np.float32)   
    if X.shape[1]>dim_threshold:
        X=preprocess_X(X)
        
    X = X.astype(np.float32)
    X = scale(X)
    # Subsampling X and y
    rng = np.random.default_rng(seed=seed)
    sample_size = min(X.shape[0], sample_size) # prevent overflow
    indices = rng.choice(np.arange(X.shape[0]), size=sample_size, replace=False)
    X, y = X[indices], y[indices]

    # Perform standard evaluation
    skf = StratifiedKFold(n_splits=n_splits)
    sum_acc = 0
    max_acc = n_splits
    for train_index, test_index in skf.split(X, y):
        feature_map_nystroem = Nystroem(gamma=1/(X.var()*X.shape[1]), n_components=300)
        data_transformed = feature_map_nystroem.fit_transform(X[train_index])
        clf = LinearSVC(tol=1e-5, dual=True, **kwargs)
        clf.fit(data_transformed, y[train_index])
        test_transformed = feature_map_nystroem.transform(X[test_index])
        acc = clf.score(test_transformed, y[test_index])
        sum_acc += acc
    avg_acc = sum_acc/max_acc
    return avg_acc


def random_triplet_eval(X, X_new, num_triplets=5):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An triplet satisfaction score is calculated by evaluating how many randomly
    selected triplets have been violated. Each point will generate 5 triplets.
    Input:
        X: A numpy array with the shape [N, p]. The higher dimension embedding
           of some dataset. Expected to have some clusters.
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset. Used to identify clusters
    Output:
        acc: The score generated by the algorithm.
    '''
    X.astype(np.float32)   
    if X.shape[1]>dim_threshold:
        X=preprocess_X(X)
        
    # Sampling Triplets
    # Five triplet per point
    anchors = np.arange(X.shape[0])
    rng = default_rng()
    triplets = rng.choice(anchors, (X.shape[0], num_triplets, 2))
    triplet_labels = np.zeros((X.shape[0], num_triplets))
    anchors = anchors.reshape((-1, 1, 1))
    
    # Calculate the distances and generate labels
    b = np.broadcast(anchors, triplets)
    distances = np.empty(b.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u,v) in b]
    labels = distances[:, :, 0] < distances[: , :, 1]
    
    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    distances_l = np.empty(b.shape)
    distances_l.flat = [np.linalg.norm(X_new[u] - X_new[v]) for (u,v) in b]
    pred_vals = distances_l[:, :, 0] < distances_l[:, :, 1]

    # Compare the labels and return the accuracy
    correct = np.sum(pred_vals == labels)
    acc = correct/X.shape[0]/num_triplets
    return acc


def neighbor_kept_ratio_eval(X, X_new, n_neighbors=30):
    '''
    This is a function that evaluates the local structure preservation.
    A nearest neighbor set is constructed on both the high dimensional space and
    the low dimensional space.
    Input:
        X: A numpy array with the shape [N, p]. The higher dimension embedding
           of some dataset. Expected to have some clusters.
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
    Output:
        acc: The score generated by the algorithm.

    '''
    X.astype(np.float32)   
    if X.shape[1]>dim_threshold:
        X=preprocess_X(X)
        
    n, dim = X.shape
    n_neighbors = min(n_neighbors, n - 1)
    nbrs_hd = construct_neighbors(X, n_neighbors=n_neighbors)
    nbrs_ld = construct_neighbors(X_new, n_neighbors=n_neighbors)
    
    same_neighbor_num=0
    
    for i in range(n):
        same_beighbor_list=np.intersect1d(nbrs_hd[i,:],nbrs_ld[i,:])
        same_neighbor_num += same_beighbor_list.shape[0]
    
    neighbor_kept_ratio = same_neighbor_num / nbrs_hd.shape[0] / nbrs_hd.shape[1]
    
    return neighbor_kept_ratio

def neighbor_notwrong_ratio_eval(X, X_new, n_neighbors=30):
    '''
    This is a function that evaluates the local structure preservation.
    A nearest neighbor set is constructed on both the high dimensional space and
    the low dimensional space.
    Input:
        X: A numpy array with the shape [N, p]. The higher dimension embedding
           of some dataset. Expected to have some clusters.
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
    Output:
        acc: The score generated by the algorithm.

    '''
    X.astype(np.float32)   
    if X.shape[1]>dim_threshold:
        X=preprocess_X(X)
        
    n, dim = X.shape
    n_neighbors = min(n_neighbors, n - 1)
    nbrs_hd = construct_neighbors(X, n_neighbors=n_neighbors)
    nbrs_ld = construct_neighbors(X_new, n_neighbors=n_neighbors)
    
    wrong_neighbor_num=0
    
    for i in range(n):
        same_beighbor_list=np.intersect1d(nbrs_hd[i,:],nbrs_ld[i,:])
        if same_beighbor_list.shape[0] < 0.5*n_neighbors:
            wrong_neighbor_num += 1
    
    neighbor_notwrong_ratio = 1-wrong_neighbor_num / nbrs_hd.shape[0] 
    
    return neighbor_notwrong_ratio

def calculate_neighbors(X, i, n_neighbors):
    '''A helper function that calculates the neighbor of a sample in a dataset.
    '''
    if isinstance(i, int):
        diff_mat = X - X[i]
    else:
        diff_mat = X - i # In this case, i is an instance of sample
    # print(f"Shape of the diff matrix is {diff_mat.shape}")
    diff_mat = np.linalg.norm(diff_mat, axis=1)
    diff_mat = diff_mat.reshape(-1)
    # Find the top n_neighbors + 1 entries
    index_list = np.argpartition(diff_mat, n_neighbors + 1)[:n_neighbors+2]
    return index_list

def spearman_correlation_eval(X, X_new, n_points=1000, random_seed=100):
    '''Evaluate the global structure of an embedding via spearman correlation in
    distance matrix, following https://www.nature.com/articles/s41467-019-13056-x
    '''
    X.astype(np.float32)   
    if X.shape[1]>dim_threshold:
        X=preprocess_X(X)
    
    # Fix the random seed to ensure reproducability
    rng = np.random.default_rng(seed=random_seed)
    dataset_size = X.shape[0]

    # Sample n_points points from the dataset randomly
    sample_index = rng.choice(np.arange(dataset_size), size=n_points, replace=False)

    # Generate the distance matrix in high dim and low dim
    dist_high = distance_matrix(X[sample_index], X[sample_index])
    dist_low = distance_matrix(X_new[sample_index], X_new[sample_index])
    dist_high = dist_high.reshape([-1])
    dist_low = dist_low.reshape([-1])

    # Calculate the correlation
    corr, pval = scipy.stats.spearmanr(dist_high, dist_low)
#    return dist_high, dist_low, corr, pval
    return corr

def centroid_knn_eval(X, X_new, y):
    '''Evaluate the global structure of an embedding via the KNC metric:
    neighborhood preservation for cluster centroids, following 
    https://www.nature.com/articles/s41467-019-13056-x
    '''
    
    X.astype(np.float32)   
    if X.shape[1]>dim_threshold:
        X=preprocess_X(X)
        
    # Calculating the cluster centers
    cluster_mean_ori, cluster_mean_new = [], []
    categories = np.unique(y)
    num_cat = len(categories)
    k= int((num_cat+2)/4)
    cluster_mean_ori = np.zeros((num_cat, X.shape[1]))
    cluster_mean_new = np.zeros((num_cat, X_new.shape[1]))
    cnt_ori = np.zeros(num_cat) # number of instances for each class

    # Only loop through the whole dataset once
    for i in range(X.shape[0]):
        ylabel = np.where(categories == int(y[i]))[0][0]
        cluster_mean_ori[ylabel] += X[i]
        cluster_mean_new[ylabel] += X_new[i]
        cnt_ori[ylabel] += 1
    cluster_mean_ori = ((cluster_mean_ori.T)/cnt_ori).T
    cluster_mean_new = ((cluster_mean_new.T)/cnt_ori).T

    # Generate the nearest neighbor list in the high dimension
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(cluster_mean_ori)
    _, indices = nbrs.kneighbors(cluster_mean_ori)
    indices = indices[:,1:] # Remove the center itself

    # Now for the low dimension
    nbr_low = NearestNeighbors(n_neighbors=k+1).fit(cluster_mean_new)
    _, indices_low = nbr_low.kneighbors(cluster_mean_new)
    indices_low = indices_low[:,1:] # Remove the center itself

    # Calculate the intersection of both lists
    len_neighbor_list = k * num_cat
    both_nbrs = 0

    # for each category, check each of its indices
    for i in range(num_cat):
        for j in range(k):
            if indices[i, j] in indices_low[i, :]:
                both_nbrs += 1
    # Compare both lists and generate the accuracy
    return both_nbrs/len_neighbor_list

def centroid_corr_eval(X, X_new, y):
    '''Evaluate the global structure of an embedding via the KNC metric:
    neighborhood preservation for cluster centroids, following 
    https://www.nature.com/articles/s41467-019-13056-x
    '''
    X.astype(np.float32)   
    if X.shape[1]>dim_threshold:
        X=preprocess_X(X)
        
    
    # Calculating the cluster centers
    cluster_mean_ori, cluster_mean_new = [], []
    categories = np.unique(y)
    num_cat = len(categories)
    cluster_mean_ori = np.zeros((num_cat, X.shape[1]))
    cluster_mean_new = np.zeros((num_cat, X_new.shape[1]))
    cnt_ori = np.zeros(num_cat) # number of instances for each class

    # Only loop through the whole dataset once
    for i in range(X.shape[0]):
        ylabel = np.where(categories == int(y[i]))[0][0]
        cluster_mean_ori[ylabel] += X[i]
        cluster_mean_new[ylabel] += X_new[i]
        cnt_ori[ylabel] += 1
    cluster_mean_ori = ((cluster_mean_ori.T)/cnt_ori).T
    cluster_mean_new = ((cluster_mean_new.T)/cnt_ori).T
    # Generate the distance matrix in high dim and low dim
    dist_high = distance_matrix(cluster_mean_ori, cluster_mean_ori)
    dist_low = distance_matrix(cluster_mean_new, cluster_mean_new)
    dist_high = dist_high.reshape([-1])
    dist_low = dist_low.reshape([-1])

    # Calculate the correlation
    corr, pval = scipy.stats.spearmanr(dist_high, dist_low)
    # return dist_high, dist_low, corr, pval
    return corr

def cluster_ratio_eval(X, X_new, max_cluster=30):
    X=X.astype(np.float32)
    X_new=X_new.astype(np.float32)
    optimalk = OptimalK(
        n_jobs=-1,
        parallel_backend="joblib")
    n_clusters_X = optimalk(X, n_refs=3, cluster_array=np.arange(1, max_cluster))
    n_clusters_X_new = optimalk(X_new, n_refs=3, cluster_array=np.arange(1, max_cluster))
    
    cluster_ratio=1/np.exp(np.absolute(n_clusters_X-n_clusters_X_new))
    
    return cluster_ratio

def cluster_ratio_eval1(X, X_new, y, labels_contineous=False):
    X=X.astype(np.float32)
    X_new=X_new.astype(np.float32)
    n_data=X.shape[0]
    min_samples=int(0.01*n_data)
    if labels_contineous:
        n_clusters_X=1
    else:
        n_clusters_X = len(set(y))
    
    
    clustering_X_new = OPTICS(min_samples=min_samples).fit(X_new)

    n_clusters_X_new = len(set(clustering_X_new.labels_))
    
    cluster_ratio=1/np.exp(0.1*np.absolute(n_clusters_X-n_clusters_X_new))
    
    return cluster_ratio

def coranking_auc_eval(X, X_new, n_neighbors=10):
    X=X.astype(np.float32)
    X_new=X_new.astype(np.float32)
    Q_coranking = compute_coranking_matrix(data_ld=X_new, data_hd = X)
    
    # Q = coranking.coranking_matrix(X, X_new)
    coranking_auc=rnx_auc_crm(Q_coranking)
    coranking_trust= trustworthiness(Q_coranking, min_k=n_neighbors, max_k=n_neighbors+1)
    conranking_cont = continuity(Q_coranking, min_k=n_neighbors, max_k=n_neighbors+1)
    conranking_lcmc = LCMC(Q_coranking, min_k=n_neighbors, max_k=n_neighbors+1)
    return coranking_auc,coranking_trust,conranking_cont,conranking_lcmc 

def curvature_simi_eval(X, X_new, n_neighbors=10):
    X=X.astype(np.float32)
    X_new=X_new.astype(np.float32)
    
    if X.shape[1]>dim_threshold:
        X=preprocess_X(X)
        
    n, dim = X.shape
    n_neighbors = min(n_neighbors, n - 1)
    nbrs_hd = construct_neighbors(X, n_neighbors=n_neighbors)
    avg_coor_X=avg_coor_compu(X, nbrs_hd)
    w_curvature_X=curv_nb(X, avg_coor_X, nbrs_hd)
    avg_curvature_X=np.absolute(np.mean(w_curvature_X))    
    avg_coor_X_new=avg_coor_compu(X_new, nbrs_hd)
    w_curvature_X_new=curv_nb(X_new, avg_coor_X_new, nbrs_hd)
    avg_curvature_X_new=np.absolute(np.mean(w_curvature_X_new))  
    # curvature_simi_eval=1.0-np.absolute(avg_curvature_X-avg_curvature_X_new)/np.max([avg_curvature_X, avg_curvature_X_new])
    curvature_simi_eval=1/np.exp(np.absolute(avg_curvature_X-avg_curvature_X_new))

    return curvature_simi_eval