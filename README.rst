Curvature Augmented Manifold Embedding and Learning -- CAMEL
=======================================

CAMEL (/CAMEL-Learn/) is a Python tool for dimension reduction and data visualization. CAMEL can perform unsupervised learning, supervised learning, semi-supervised learning, metric learning, and inverse learning! CAMEL offers a simple and intuitive API.


========================================

Install:

pip install camel-learn


Simple code example: (more coming)


import matplotlib.pyplot as plt
import time
from camel import CAMEL
from sklearn import datasets


t1=time.time()

X, y = datasets.make_swiss_roll(n_samples=10000, random_state=None)

reducer= CAMEL()

X_embedding = reducer.fit_transform(X)


print(time.time()-t1)

y = y.astype(int) #convert to category for easy visulization

# Visualization
# MNIST visualization
plt.figure(1)
plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c=y, cmap='jet', s=0.2)
plt.title('CAMEL Embedding')
plt.tight_layout()
plt.show()

