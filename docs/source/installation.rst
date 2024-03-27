Installation
=============

.. _installation:

Installation
------------

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

**Install Options**

.. code:: console

     (.venv) pip install camel-learn

If pip is having difficulties pulling the dependencies, then I'd suggest installing
the dependencies manually. If you have installed older versions, you can upgrade CAMEL by

.. code:: console

     (.venv) pip install camel-learn --upgrade


If yo uare using Anaconda, please open anaconda-navigator first. On the left panel, select "enviroments". Click on the enviroment you are using and open 
a terminal. pip install or pip upgrade camel-learn using the code above. re-start ananconda-navigator to make the modulues up-to-date and available in your enviroments.
