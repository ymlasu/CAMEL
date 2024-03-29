# need to install the following packages
import umap


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
output_path = "../output/inverse_learning/"


methods_compare= ['CAMEL']
data_compare = ['MNIST']

X, y = data_prep(data_path, data_compare[0], size=10000)
    
if len(set(y))>0.1*y.shape[0]:
    labels_contineous=True
    target_type='numerical'
    target_metric='l2'
else:
    labels_contineous=False
    target_type='categorical'
    target_metric='categorical'
            
            
reducer= CAMEL(target_type=target_type, random_state=1)

X_embedding = reducer.fit_transform(X)
y = y.astype(int)

#Xnewdata=X[1:X.shape[0],:]
#X_transformed = reducer.transform(Xnew,basis=X)

plt.figure(figsize=(6,6),layout='constrained',dpi=300)
plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c=y, cmap='jet', s=0.2)
plt.title('CAMEL Embedding')

plt.tight_layout()
plt.show()







corners = np.array([
    [-8, 15],  # top left corner
    [+20, 0.0], #top right corner
    [-18, -5],  # bottom left corner
    [+0, -15],  # bottom right
])

test_pts = np.array([
    (corners[0]*(1-x) + corners[1]*x)*(1-y) +
    (corners[2]*(1-x) + corners[3]*x)*y
    for y in np.linspace(0, 1, 10)
    for x in np.linspace(0, 1, 10)
])

# X_embedding_inverse = reducer.inverse_transform(test_pts)
X_embedding_inverse = reducer.inverse_transform(test_pts, init='random')

# Set up the grid
fig = plt.figure(figsize=(24,12),layout='constrained',dpi=300)
gs = GridSpec(10, 20, fig)
scatter_ax = fig.add_subplot(gs[:, :10])
digit_axes = np.zeros((10, 10), dtype=object)
for i in range(10):
    for j in range(10):
        digit_axes[i, j] = fig.add_subplot(gs[i, 10 + j])

# Use umap.plot to plot to the major axis
# umap.plot.points(mapper, labels=labels, ax=scatter_ax)
scatter_ax.scatter(X_embedding[:, 0], X_embedding[:, 1],
                    c=y, cmap='jet', s=5)
scatter_ax.set(xticks=[], yticks=[])

# Plot the locations of the text points
scatter_ax.scatter(test_pts[:, 0], test_pts[:, 1], marker='x', c='k', s=100)

# Plot each of the generated digit images
for i in range(10):
    for j in range(10):
        digit_axes[i, j].imshow(X_embedding_inverse[i*10 + j].reshape(28, 28))
        digit_axes[i, j].set(xticks=[], yticks=[])
        
        
        
plt.savefig(output_path +'camel_inverse_learning_MNIST_random.png')

# X_embedding_inverse = reducer.inverse_transform(test_pts)
X_embedding_inverse = reducer.inverse_transform(test_pts, init='interpolate')

# Set up the grid
fig = plt.figure(figsize=(24,12),layout='constrained',dpi=300)
gs = GridSpec(10, 20, fig)
scatter_ax = fig.add_subplot(gs[:, :10])
digit_axes = np.zeros((10, 10), dtype=object)
for i in range(10):
    for j in range(10):
        digit_axes[i, j] = fig.add_subplot(gs[i, 10 + j])

# Use umap.plot to plot to the major axis
# umap.plot.points(mapper, labels=labels, ax=scatter_ax)
scatter_ax.scatter(X_embedding[:, 0], X_embedding[:, 1],
                    c=y, cmap='jet', s=5)
scatter_ax.set(xticks=[], yticks=[])

# Plot the locations of the text points
scatter_ax.scatter(test_pts[:, 0], test_pts[:, 1], marker='x', c='k', s=100)

# Plot each of the generated digit images
for i in range(10):
    for j in range(10):
        digit_axes[i, j].imshow(X_embedding_inverse[i*10 + j].reshape(28, 28))
        digit_axes[i, j].set(xticks=[], yticks=[])
        
        
        
plt.savefig(output_path +'camel_inverse_learning_MNIST_interpolate.png')


######################################################################################


methods_compare= ['CAMEL']
data_compare = ['FMNIST']

X, y = data_prep(data_path, data_compare[0], size=10000)

if len(set(y))>0.1*y.shape[0]:
    labels_contineous=True
    target_type='numerical'
    target_metric='l2'
else:
    labels_contineous=False
    target_type='categorical'
    target_metric='categorical'


reducer= CAMEL(target_type=target_type, random_state=1)

X_embedding = reducer.fit_transform(X, y)
y = y.astype(int)

#Xnewdata=X[1:X.shape[0],:]
#X_transformed = reducer.transform(Xnew,basis=X)

plt.figure(figsize=(6,6),layout='constrained',dpi=300)
plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c=y, cmap='jet', s=0.2)
plt.title('CAMEL Embedding')

plt.tight_layout()
plt.show()







corners = np.array([
    [-15, 20],  # top left corner
    [+15, 20], #top right corner
    [-15, -20],  # bottom left corner
    [+15, -20],  # bottom right
])

test_pts = np.array([
    (corners[0]*(1-x) + corners[1]*x)*(1-y) +
    (corners[2]*(1-x) + corners[3]*x)*y
    for y in np.linspace(0, 1, 10)
    for x in np.linspace(0, 1, 10)
])

# X_embedding_inverse = reducer.inverse_transform(test_pts)
X_embedding_inverse = reducer.inverse_transform(test_pts)

# Set up the grid
fig = plt.figure(figsize=(24,12),layout='constrained',dpi=300)
gs = GridSpec(10, 20, fig)
scatter_ax = fig.add_subplot(gs[:, :10])
digit_axes = np.zeros((10, 10), dtype=object)
for i in range(10):
    for j in range(10):
        digit_axes[i, j] = fig.add_subplot(gs[i, 10 + j])

# Use umap.plot to plot to the major axis
# umap.plot.points(mapper, labels=labels, ax=scatter_ax)
scatter_ax.scatter(X_embedding[:, 0], X_embedding[:, 1],
                    c=y, cmap='jet', s=5)
scatter_ax.set(xticks=[], yticks=[])

# Plot the locations of the text points
scatter_ax.scatter(test_pts[:, 0], test_pts[:, 1], marker='x', c='k', s=100)

# Plot each of the generated digit images
for i in range(10):
    for j in range(10):
        digit_axes[i, j].imshow(X_embedding_inverse[i*10 + j].reshape(28, 28))
        digit_axes[i, j].set(xticks=[], yticks=[])



plt.savefig(output_path +'camel_inverse_learning_FMNIST.png')


##########################################################################################
methods_compare= ['UMAP']
data_compare = ['FMNIST']

X, y = data_prep(data_path, data_compare[0], size=10000)
        
if len(set(y))>0.1*y.shape[0]:
    labels_contineous=True
    target_type='numerical'
    target_metric='l2'
else:
    labels_contineous=False
    target_type='categorical'
    target_metric='categorical'
            
            
reducer= umap.UMAP(target_metric=target_metric, random_state=1)

X_embedding = reducer.fit_transform(X, y)
y = y.astype(int)

#Xnewdata=X[1:X.shape[0],:]
#X_transformed = reducer.transform(Xnew,basis=X)

plt.figure(figsize=(6,6),layout='constrained',dpi=300)
plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c=y, cmap='jet', s=0.2)
plt.title('CAMEL Embedding')

plt.tight_layout()
plt.show()







corners = np.array([
    [-15, 20],  # top left corner
    [+30, 20], #top right corner
    [-15, -10],  # bottom left corner
    [+30, -10],  # bottom right
])

test_pts = np.array([
    (corners[0]*(1-x) + corners[1]*x)*(1-y) +
    (corners[2]*(1-x) + corners[3]*x)*y
    for y in np.linspace(0, 1, 10)
    for x in np.linspace(0, 1, 10)
])

# X_embedding_inverse = reducer.inverse_transform(test_pts)
X_embedding_inverse = reducer.inverse_transform(test_pts)

# Set up the grid
fig = plt.figure(figsize=(24,12),layout='constrained',dpi=300)
gs = GridSpec(10, 20, fig)
scatter_ax = fig.add_subplot(gs[:, :10])
digit_axes = np.zeros((10, 10), dtype=object)
for i in range(10):
    for j in range(10):
        digit_axes[i, j] = fig.add_subplot(gs[i, 10 + j])

# Use umap.plot to plot to the major axis
# umap.plot.points(mapper, labels=labels, ax=scatter_ax)
scatter_ax.scatter(X_embedding[:, 0], X_embedding[:, 1],
                    c=y, cmap='jet', s=5)
scatter_ax.set(xticks=[], yticks=[])

# Plot the locations of the text points
scatter_ax.scatter(test_pts[:, 0], test_pts[:, 1], marker='x', c='k', s=100)

# Plot each of the generated digit images
for i in range(10):
    for j in range(10):
        digit_axes[i, j].imshow(X_embedding_inverse[i*10 + j].reshape(28, 28))
        digit_axes[i, j].set(xticks=[], yticks=[])
        
        
        
plt.savefig(output_path +'umap_inverse_learning_FMNIST.png')










