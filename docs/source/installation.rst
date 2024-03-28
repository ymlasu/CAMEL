Installation
=============

.. _installation:

Required packages
------------------

CAMEL Requirements:

* Python 3.6 or greater
* numpy
* scikit-learn
* numba
* annoy
* pandas

Recommended packages:

* For plotting
   * matplotlib
* For metrics evaluation
   * gap statistics, coranking, optics
* For comaprision with other methods
   * umap-learn, trimap, pacmap

Install Options
--------------------

.. code:: console

   pip install camel-learn

If pip is having difficulties pulling the dependencies, then I'd suggest installing
the dependencies manually. If you have installed older versions, you can upgrade CAMEL by

.. code:: console

   pip install camel-learn --upgrade


If you are using Anaconda, please open anaconda-navigator first. On the left panel, select "enviroments". Click on the enviroment you are using and open 
a terminal. pip install or pip upgrade camel-learn using the code above. re-start ananconda-navigator to make the modulues up-to-date and available in your enviroments.


Alternatively, you can download the git repo to a local drive, and import the CAMEL class from the source code folder in the github repo. You do need to install other required packages.

