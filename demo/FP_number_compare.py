# need to install the following packages

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from camel import CAMEL
from eval_metrics import *

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
output_path = "../output/far points/"

# methods_compare= ['TSNE', 'UMAP', 'TriMAP', 'PaCMAP', 'CAMEL']
# data_compare = ['swiss_roll', 'MNIST']



methods_compare= ['CAMEL']
data_compare = ['swiss_roll', 'mammoth','MNIST', 'FMNIST', 'USPS', '20NG', 'coil_20', 'coil_100']
para_list = np.array([5, 10, 40, 80])

# methods_compare= ['CAMEL']
# data_compare = ['swiss_roll']

n_monte=1

n_methods=len(methods_compare)
n_data=len(data_compare)
n_para=len(para_list)

# metrics_knn=np.zeros([n_monte,n_data,n_methods])
# metrics_svm=np.zeros([n_monte,n_data,n_methods])
# metrics_triplet=np.zeros([n_monte,n_data,n_methods])
# metrics_nkr=np.zeros([n_monte,n_data,n_methods])
# metrics_scorr=np.zeros([n_monte,n_data,n_methods])
# metrics_cenknn=np.zeros([n_monte,n_data,n_methods])
# metrics_cencorr=np.zeros([n_monte,n_data,n_methods])
# metrics_clusterratio=np.zeros([n_monte,n_data,n_methods])
# metrics_coranking_auc=np.zeros([n_monte,n_data,n_methods])
# metrics_coranking_trust=np.zeros([n_monte,n_data,n_methods])
# metrics_coranking_cont=np.zeros([n_monte,n_data,n_methods])
# metrics_coranking_lcmc=np.zeros([n_monte,n_data,n_methods])
metrics_curvature_simi=np.zeros([n_monte,n_data,n_para])
# metrics_nnwr=np.zeros([n_monte,n_data,n_methods])

# Set up the grid
fig = plt.figure(figsize=(8*n_para,6*n_data*n_methods),layout='constrained',dpi=300)
gs = GridSpec(n_data*n_methods, n_para, figure=fig)

digit_axes = np.zeros((n_data*n_methods, n_para), dtype=object)


for k in range(n_monte):
    
    # if methods_compare[k] == 'PaCMAP':
    #     transformer = pacmap.PaCMAP()
    # elif methods_compare[k]  == 'UMAP':
    #     transformer = umap.UMAP()
    # elif methods_compare[k] == 'TSNE':
    #     transformer = TSNE()
    # elif methods_compare[k]  == 'TriMAP':
    #     transformer = trimap.TRIMAP()
    # elif methods_compare[k]  == 'CAMEL':
    #     transformer = CAMEL(n_neighbors=10, FP_number=20, w_neighbors=1.0, 
    #                         tail_coe=0.05, w_curv=0.1, w_FP=20, num_iters=400, target_weight=weight_list[j], random_state=None)            
    # else:
    #     print("Incorrect method specified")
    #     assert(False)

    for i in range(n_data):
        X, y = data_prep(data_path, data_compare[i], size=10000)
        if len(set(y))>0.1*y.shape[0]:
            labels_contineous=True
            target_type='numerical'
            target_metric='l2'
        else:
            labels_contineous=False
            target_type='categorical'
            target_metric='categorical'
        for j in range(n_para):
            
            transformer = CAMEL( FP_number=int(para_list[j]), target_type=target_type, random_state=1)     

            X_embedding = transformer.fit_transform(X)


            if k == 0:

                y_plot = np.copy(y).astype(int)
        
                # Visualization
                
    
                
                digit_axes[k*n_data+i, j] = fig.add_subplot(gs[k*n_data+i, j])
                digit_axes[k*n_data+i, j].scatter(X_embedding[:, 0], X_embedding[:, 1],
                                    c=y_plot, cmap='jet', s=0.2)
                title_embedding = 'Neighbor Number: '+ str(para_list[j])
                digit_axes[k*n_data+i, j].set_title(title_embedding,fontsize=12)
                digit_axes[k*n_data+i, j].set_axis_off()
            
            # plt.show()
    
            
            

            metrics_curvature_simi[k,i,j] = curvature_simi_eval(X, X_embedding)

avg_curvature_simi=np.mean(metrics_curvature_simi, axis=0)
std_curvature_simi=np.std(metrics_curvature_simi, axis=0)

x = np.arange(len(data_compare))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', dpi=300)

for i in range(avg_curvature_simi.shape[1]):
    offset = width * multiplier
    rects = ax.bar(x + offset, avg_curvature_simi[:,i], width, yerr=std_curvature_simi[:,i],label='FP Number: ' + str(para_list[i]))
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Curvature Similarity Score', fontsize=14)
ax.set_title('Effect of FP number',fontsize=16)
ax.set_xticks(x + width, data_compare)
ax.set_xticklabels(data_compare, fontsize=14, rotation = 45)
ax.legend(loc='upper left', ncols=2, fontsize=10)
ax.set_ylim(0, 1.5)

plt.show()




np.save(output_path + '/metrics_curvature_simi', metrics_curvature_simi)


plt.savefig(output_path +'model_compare.png')
        

 