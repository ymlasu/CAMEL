.. -*- mode: rst -*-

.. image:: docs/Camel_logo.png
  :width: 600
  :alt: CAMELlogo
  :align: center

|pypi_version|_ |pypi_downloads|_

.. |pypi_version| image:: https://img.shields.io/pypi/v/camel-learn.svg
.. _pypi_version: https://pypi.python.org/pypi/camel-learn/


Curvature Augmented Manifold Embedding and Learning -- CAMEL
=======================================

CAMEL (/CAMEL-Learn/) is a Python tool for dimension reduction and data visualization. CAMEL can perform unsupervised learning, supervised learning, semi-supervised learning, metric learning, and inverse learning! CAMEL offers a simple and intuitive API.
=====
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


