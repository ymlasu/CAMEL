.. -*- mode: rst -*-

.. image:: docs/Camel_logo.png
  :width: 600
  :alt: CAMELlogo
  :align: center

|pypi_version|_ 

.. |pypi_version| image:: https://img.shields.io/pypi/v/camel-learn.svg
.. _pypi_version: https://pypi.python.org/pypi/camel-learn/


Curvature Augmented Manifold Embedding and Learning -- CAMEL
=============

CAMEL is a Python tool for dimension reduction and data visualization. CAMEL can perform unsupervised learning, supervised learning, semi-supervised learning, metric learning, and inverse learning! CAMEL offers a simple and intuitive API.
----------
Installing
----------

CAMEL Requirements:

* Python 3.6 or greater
* numpy
* scikit-learn
* numba
* annoy

Recommended packages:

* For plotting
   * matplotlib
* For metrics evaluation
   * gap statistics, coranking, optics

**Install Options**

.. code:: bash

     pip install camel-learn

If pip is having difficulties pulling the dependencies then we'd suggest installing
the dependencies manually using anaconda. The author has tried Anaconda in Mac Os 14 with M1 and M2 cpu.


======

---------------
How to use CAMEL
---------------

The camel package is inspired and developed based on many dimention reduction pakckages, such UMAP, TriMAP, and PaCMAP, which follow the similar setting from sklearn classes. Thus, CAMEL shares the similar calling format using the CAMEL API.

1. There is only one class CAMEL().
2. fit(X, y) and fit_transform(X, y) perform training of embedding of data and construct a "model". X refers to input feature data and y refers to input label data. y is optional and can also has missing/NaN data. This module is mainly used for unsupetvised, supervised , and semi_supervised learning.
3. transform(Xnew, basis) is for embedding if new testing data Xnew is provided and the model is constructed using basis datasets. basis data is optional. This module is mainly used for metric learning where the metric model is already learned from training data, weather it is in supervised, unsupervised, or semi_supervised learning. 
4. invser_transform(ynews, X, y) is used for inverse embedding and dimension augmentation from low dimension to high dimension. This module assumes that you have a forward embedding constructed from training data X (basis feature) and y (embedding of basis feature). Then, one can reverse this process by construct a feature speace vector from a new unseen point in low dimension point ynew. This is in analogy to genrative model from a latent space in ML. 

The CAMEL is very easy to start with. you can start a basic unsupetvised learning job and plotting with less than 10 lines of code!

.. code:: python

    import matplotlib.pyplot as plt
    from camel import CAMEL
    from sklearn import datasets

    X, y = datasets.make_swiss_roll(n_samples=10000, random_state=None)

    reducer= CAMEL()

    X_embedding = reducer.fit_transform(X)

    y = y.astype(int) #convert to category for easy visulization

    # Visualization

    plt.figure(1)
    plt.scatter(X_embedding[:, 0], X_embedding[:, 1], c=y, cmap='jet', s=0.2)
    plt.title('CAMEL Embedding')
    plt.tight_layout()
    plt.show()


Simple code examples in test folder: (more coming)

=====
API
=====
There are several parameters that can control the results and performance of the CAMEL. Default values have been set if you want to start quickly. If you want to fine tune the CAMEL, below is a description of several main factors.

- ''n_components'': int, default=2
        Dimensions of the embedded space. Typicalvalues are 2 or 3. it can be any integer.

- ' ' n_neighbors'': int, default=10
        Number of neighbors considered for nearest neighbor pairs for local structure preservation.

- ''FP_number'': float, default=20
        Number of further points(e.g. 20 Further pairs per node)
        Further pairs are used for both local and global structure preservation.

- ''tail_coe'': float, default=0.05
        Parameter to control the attractive force of neighbors (1/(1+tail_coe*dij)**2), smaller values indicate flat tail, do not recommend ro change
    
- ''w_neighbors'': float, default=1.0
        weight coefficient for attractive force of neighbors, large values indicates strong force for the same distance metric
        
- ''w_curv'': float, default=0.001
        weight coefficient for attractive/repulsive force due to local curvature, large values indicates strong force for the same distance metric        

- ''w_FP'': float, default=20
        weight coefficient for repulsive force of far points, large values indicates strong force for the same distance metric    
    
- ''lr'': float, default=1.0
        Learning rate of the Adam optimizer for embedding. donot recommend to change.

- ''num_iters'': int, default=400
        Number of iterations for the optimization of embedding. I observe that 200 is sufficient for most cases and 400 is used here for safe reason.

- ''target_weight'': float, default=0.5
        weight factor for target/label during the supervised learning, 0 indicates no weight and it reduces to unsupervised one,
        1 indicates infinity weight (set as a large value in practice.

- ''random_state'': int, optional
        Random state for the camel instance.
        Setting random state is useful for repeatability.



other setting can be seen in the source code and will be updated i fututre documentation.



Theory and Reference
---------
The theory behind the CAMEL is deing submitted and reviewed. I am on a flight typing on a phone. Thus, I am only putting the draft pdf file for interested readers to take a look. will update this when I return.

.. image:: docs/Camel_learn.pdf


Happy CAMEL!! :-)



