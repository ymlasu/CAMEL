# need to install the following packages
import umap
import trimap
import pacmap

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from camel import CAMEL
from eval_metrics import *

from time import time

from sklearn.manifold import TSNE
#from pacmap import PaCMAP
from sklearn.datasets import make_swiss_roll, make_s_curve

global _RANDOM_STATE
_RANDOM_STATE = None


def data_prep(data_path, dataset='MNIST', size=10000):
    '''
    This function loads the dataset as numpy array.
    Input:
        data_path: path of the folder you store all the data needed.
        dataset: the name of the dataset.
        size: the size of the dataset. This is useful when you only
              want to pick a subset of the data
    Output:
        X: the dataset in numpy array
        labels: the labels of the dataset.
    '''

    if dataset == 'MNIST':
        X = np.load(data_path + '/mnist_images.npy', allow_pickle=True).reshape(70000, 28*28)
        labels = np.load(data_path + '/mnist_labels.npy', allow_pickle=True)
    elif dataset == 'FMNIST':
        X = np.load(data_path + '/fmnist_images.npy', allow_pickle=True).reshape(70000, 28*28)
        labels = np.load(data_path + '/fmnist_labels.npy', allow_pickle=True)
    elif dataset == 'coil_20':
        X = np.load(data_path + '/coil_20.npy', allow_pickle=True).reshape(1440, 128*128)
        labels = np.load(data_path + '/coil_20_labels.npy', allow_pickle=True)
    elif dataset == 'coil_100':
        X = np.load(data_path + '/coil_100.npy', allow_pickle=True).reshape(7200, -1)
        labels = np.load(data_path + '/coil_100_labels.npy', allow_pickle=True)
    elif dataset == 'mammoth':
        with open(data_path + '/mammoth_umap.json', 'r') as f:
            labels = json.load(f)
        X= labels['3d']
        X = np.array(X)
        labels = labels['labels']
        labels = np.array(labels)
    elif dataset == '20NG':
        X = np.load(data_path + '/20NG.npy', allow_pickle=True)
        labels = np.load(data_path + '/20NG_labels.npy', allow_pickle=True)
    elif dataset == 'USPS':
        X = np.load(data_path + '/USPS.npy', allow_pickle=True)
        labels = np.load(data_path + '/USPS_labels.npy', allow_pickle=True)
    elif dataset == 'CIFAR_10':
        X = pd.read_csv(data_path + '/CIFAR_10.csv')
        X=X.to_numpy(dtype=np.float32)
        X=X[:,1:]
        labels = pd.read_csv(data_path + '/CIFAR_10_labels.csv')
        labels =labels['class']
        labels =labels.to_numpy(dtype=np.int32)
    elif dataset == 'swiss_roll':
        X, labels = make_swiss_roll(n_samples=size, random_state=_RANDOM_STATE)
    elif dataset == 's_curve':
        X, labels = make_s_curve(n_samples=size, random_state=_RANDOM_STATE)
    elif dataset == 's_curve_hole':
        X, labels = make_s_curve(n_samples=size, random_state=_RANDOM_STATE)
        anchor = np.array([0, 1, 0])
        indices = np.sum(np.square(X-anchor), axis=1) > 0.3
        X, labels = X[indices], labels[indices]
    elif dataset == 'swiss_roll_hole':
        X, labels = make_swiss_roll(n_samples=size, random_state=_RANDOM_STATE)
        anchor = np.array([-10, 10, 0])
        indices = np.sum(np.square(X-anchor), axis=1) > 20
        X, labels = X[indices], labels[indices]
    elif dataset == '2D_curve':
        x = np.arange(-5.5, 9, 0.01)
        y = 0.01 * (x + 5) * (x + 2) * (x - 2) * (x - 6) * (x - 8)
        noise = np.random.randn(x.shape[0]) * 0.01
        y += noise
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        X = np.hstack((x, y))
        labels = x
    else:
        print('Unsupported dataset')
        assert(False)
    return X[:size], labels[:size]


data_path = "../data/"
output_path = "../output/"

# methods_compare= ['CAMEL']
# data_compare = ['MNIST']


methods_compare= ['TSNE', 'UMAP', 'TriMAP', 'PaCMAP', 'CAMEL']
data_compare = ['swiss_roll', 'mammoth', 'coil_20', 'coil_100','MNIST', 'FMNIST', '20NG', 'USPS']

# methods_compare= ['CAMEL']
# data_compare = ['swiss_roll']

n_monte=1

n_methods=len(methods_compare)
n_data=len(data_compare)
total_time=np.zeros([n_monte,n_data,n_methods])
metrics_knn=np.zeros([n_monte,n_data,n_methods])
metrics_svm=np.zeros([n_monte,n_data,n_methods])
metrics_triplet=np.zeros([n_monte,n_data,n_methods])
metrics_nkr=np.zeros([n_monte,n_data,n_methods])
metrics_scorr=np.zeros([n_monte,n_data,n_methods])
metrics_cenknn=np.zeros([n_monte,n_data,n_methods])
metrics_cencorr=np.zeros([n_monte,n_data,n_methods])
metrics_clusterratio=np.zeros([n_monte,n_data,n_methods])
metrics_coranking_auc=np.zeros([n_monte,n_data,n_methods])
metrics_coranking_trust=np.zeros([n_monte,n_data,n_methods])
metrics_coranking_cont=np.zeros([n_monte,n_data,n_methods])
metrics_coranking_lcmc=np.zeros([n_monte,n_data,n_methods])
metrics_curvature_simi=np.zeros([n_monte,n_data,n_methods])
metrics_nnwr=np.zeros([n_monte,n_data,n_methods])

# Set up the grid
fig = plt.figure(figsize=(6*n_methods,6*n_data),layout='constrained',dpi=300)
gs = GridSpec(n_data, n_methods, figure=fig)
scatter_ax = fig.add_subplot(gs[:, :])
digit_axes = np.zeros((n_data, n_methods), dtype=object)
scatter_ax.set(xticks=[], yticks=[])

for k in range(n_monte):

    for i in range(n_data):
        X, y = data_prep(data_path, data_compare[i], size=10000)
        if len(set(y))>0.1*y.shape[0]:
            labels_contineous=True
        else:
            labels_contineous=False
        for j in range(n_methods):
          
            if methods_compare[j] == 'PaCMAP':
                transformer = pacmap.PaCMAP()
            elif methods_compare[j]  == 'UMAP':
                transformer = umap.UMAP()
            elif methods_compare[j] == 'TSNE':
                transformer = TSNE()
            elif methods_compare[j]  == 'TriMAP':
                transformer = trimap.TRIMAP()
            elif methods_compare[j]  == 'CAMEL':
                transformer = CAMEL(n_neighbors=10, FP_number=20, w_neighbors=1.0, 
                                    tail_coe=0.05, w_curv=0.001, w_FP=20, num_iters=400, random_state=None)            
            else:
                print("Incorrect method specified")
                assert(False)
            start_time = time()
            X_embedding = transformer.fit_transform(X)
            total_time [k,i,j] = time() - start_time
    
            y = y.astype(int)
    
            # Visualization
            
            if k == 0:
            
                digit_axes[i, j] = fig.add_subplot(gs[i, j])
                digit_axes[i, j].scatter(X_embedding[:, 0], X_embedding[:, 1],
                                    c=y, cmap='jet', s=0.2)
                title_embedding = methods_compare[j] +' ' +'Embedding of '+ data_compare[i]
                digit_axes[i, j].set_title(title_embedding,fontsize=12)
                digit_axes[i, j].set_axis_off()
            
            # plt.show()
    
            
            
            #metrics-based evaluation
            #1. knn_eval_large
            metrics_knn[k,i,j] = knn_eval_large(X_embedding, y)
            #2. metrics_svm
            metrics_svm[k,i, j] = svm_eval_large(X_embedding, y)
            #3. random triplet eval
            metrics_triplet[k,i,j] = random_triplet_eval(X, X_embedding)
            #4. neighbor kept ratio nkr  - has issues
            metrics_nkr[k,i,j] = neighbor_kept_ratio_eval(X, X_embedding)
            #5. spearman correaltion
            metrics_scorr[k,i,j] = spearman_correlation_eval(X, X_embedding)
            # #6. centroid knn 
            metrics_cenknn[k,i,j] = centroid_knn_eval(X, X_embedding, y)
            #centroid dist corr
            metrics_cencorr[k,i,j] = centroid_corr_eval(X, X_embedding, y)
            #cluster ratio
            metrics_clusterratio[k,i,j] = cluster_ratio_eval1(X, X_embedding, y, labels_contineous)
            # coranking auc 
            # metrics_coranking_auc[k,i,j], metrics_coranking_trust[k,i,j],
            # metrics_coranking_cont[k,i,j], metrics_coranking_lcmc[k,i,j]= coranking_auc_eval(X, X_embedding)
            coranking_auc,coranking_trust,conranking_cont,conranking_lcmc = coranking_auc_eval(X, X_embedding)
            metrics_coranking_auc[k,i,j]=coranking_auc
            metrics_coranking_trust[k,i,j]=coranking_trust
            metrics_coranking_cont[k,i,j]=conranking_cont
            metrics_coranking_lcmc[k,i,j]=conranking_lcmc
            # # curvature correlation 
            metrics_curvature_simi[k,i,j] = curvature_simi_eval(X, X_embedding)
            # neighbor not wrong ratio
            metrics_nnwr[k,i,j] = neighbor_notwrong_ratio_eval(X, X_embedding)




#save all results in output folder        
np.save(output_path + '/total_time.npy', total_time)
np.save(output_path + '/metrics_knn.npy', metrics_knn)
np.save(output_path + '/metrics_svm.npy', metrics_svm)
np.save(output_path + '/metrics_triplet.npy', metrics_triplet)
np.save(output_path + '/metrics_nkr.npy', metrics_nkr)
np.save(output_path + '/metrics_scorr.npy', metrics_scorr)
np.save(output_path + '/metrics_cenknn.npy', metrics_cenknn)
np.save(output_path + '/metrics_cencorr.npy', metrics_cencorr)
np.save(output_path + '/metrics_clusterratio', metrics_clusterratio)
np.save(output_path + '/metrics_coranking_auc', metrics_coranking_auc)
np.save(output_path + '/metrics_coranking_trust', metrics_coranking_trust)
np.save(output_path + '/metrics_coranking_cont', metrics_coranking_cont)
np.save(output_path + '/metrics_coranking_lcmc', metrics_coranking_lcmc)
np.save(output_path + '/metrics_curvature_simi', metrics_curvature_simi)
np.save(output_path + '/metrics_nnwr', metrics_nnwr)
np.save(output_path + '/methods_compare.npy', methods_compare)
np.save(output_path + '/data_compare.npy', data_compare)

plt.savefig(output_path +'model_compare.png')
        
print(total_time)        

        #####check MNIST inverse
 