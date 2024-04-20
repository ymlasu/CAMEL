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
the dependencies manually. For example, you need annoy package (https://pypi.org/project/annoy/) to run CAMEL for knn graph. If you did not have it installed, please 

.. code:: console

   pip install annoy

If you are running mac or linux, c++ builder should be already set. If you are running the code in windows, you may need ot install the c++ builder. Please follow this link (https://visualstudio.microsoft.com/visual-cpp-build-tools/) to install the C++ builders first.

If you wan tto run the demo code and compare with other dimention reduction methods. You also need to install UMAP, pacmap, and Trimap.

.. code:: console

   pip install umap-learn
   pip install pacmap
   pip install trimap

If you want to use the metrics in the arxiv paper to do the comparision. You need to install addtional packages for metrics defination. GapStatistics (https://pypi.org/project/gap-stat/) is needed and you also need coranking package (https://pypi.org/project/pycoranking/).

.. code:: console

   pip install gap-stat
   pip install pycoranking

If you have installed older versions, you can upgrade CAMEL by

.. code:: console

   pip install camel-learn --upgrade


If you are using Anaconda, please open anaconda-navigator first. On the left panel, select "enviroments". Click on the enviroment you are using and open 
a terminal. pip install or pip upgrade camel-learn using the code above. re-start ananconda-navigator to make the modulues up-to-date and available in your enviroments.


Alternatively, you can download the git repo to a local drive, and import the CAMEL class from the source code folder in the github repo. You do need to install other required packages.


Data Options
--------------------

The git repo has some small size demonstration data available. readme file in the data folder also has the link for other large data downloading link. You can put your own data in the folder for processing. 
