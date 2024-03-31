Git Folder Structure
=====================

CAMEL also includes some other files for demonstration and evaluation, which are not included in the pip installed modulues. They can be donwloaded
from the Github repo at https://github.com/ymlasu/CAMEL

The folder structure is schematically shown below

* -
    - data
    - demo
    - docs
    - plot_util
    - source
    - test
        * READEME.md
        * setup.cfg
        * pyproject.toml
        * .readthedocs.yaml
        * license


Dicussion for each folder is given below.

data
-----
data folder includes all used datasets in the arXiv paper (https://arxiv.org/abs/2403.14813) and some addtional datasets for CAMEL
analysis. It is meant to have a place for data storage and management, which can be accessed with the provided example code (in the demo folder)
to extract data. data folder currently has 20NG, coil_20, mammoth, and USPS data. Other large datasets can be downloaded from the links
shown in the "READEME_addtional_data_download.md" file. Most of the data is in the Python numpy format, some are in json or csv format.
Each data usually has two files: XXXX.npy and XXXX_labels.npy. XXXX refers to the name of the dataset and contains all input features of the data.
XXXX_labels.npy is for the label information of the dataset.

demo
------
demo folder includes examples used in the arXiv paper on various learning tasks and parametric studies. These Python files serves as 
template for interested readers to reproduce the results and revise them to be suitable for your own applications. Brief explaination for each files
is given below.

- eval_metrics.py: This files includes 14 quantitative evaluation metrics used inthe arXiv paper to evalue the model performance. The metrics can be called using the 
| fuction lie XXXX_XXXX_eval.py, where XXXX refers to the metrics names. Detailed usage can be found in other example files in this folder.


- FP_number_compare.py: This is a parametric study demo file for CAMEL to evalute the effect of diffrent far point (FP) numbers on the embedding quality.

- inverse_learning_compare.py: This is a template for inverse learning. THis file compares the CAMEL and UMPA performance in generating images
| for MNIST and Fashion MNIST data. This can be extended to other generative modeling from low-dimentional embedding

- metric_learning_compare.py: This is a template for metric learning, which is formulated as a projection of new data with learned embedding
| from either unsupervised and supervised learning. CAMEL, UMAP, and PaCMAP are compared here.

- model_compare.py: This is the file for basic unsupervised learning comparision of 5 methods: tSNE, UMAP, Trimap, PacMAP, and CAMEL.
| THis is a good template to see how to input data file, setup embedding, metrics evaluation, and plotting.

- neighbor_numbercompare.py: This is a parametric study demo file for CAMEL to evalute the effect of diffrent neighbor numbers on the embedding quality.

- semi_supervised_learning_compare.py: This is a template for semi supervised learning learning, which is formulated as a data augmentation and supervised learning with CAMEL.
| CAMEL and UMAP are compared here.

- supervised_learning_compare.py: This is a template for supervised learning learning, which is formulated as a knn revision with label information.
| CAMEL and UMAP are compared here.   

- weight_curvature_compare.py: This is a parametric study demo file for CAMEL to evalute the effect of diffrent weight coefficients for curvature-induced force field on the embedding quality.


docs
-----

docs folder includes all documentation for readthedocs and images and pdf files for the CAMEL project. All .rst files for the readthedocs.io are tored in the subfolder \resource.


output
-------

output folder is a folder to store the output files and images of each demo python code. It is empty in the current folder, but it is suggested when you donwload the git_folder_structure
| files to run locally, which orgnize all your outfiles for easy reproduction.



plot_util
----------

This folder inlcudes plot utility functions to generate diffrent images shown in the arXiv paper. It currently has one files for the evaluation metrics bar plot.
| Other plotting functions can be found in the demo folder files.

source
--------

This folder contains the source code of CAMEL. It also contains _init_.py to generate pypi installation wheel.


test
-----
This folder contains very simple template for CAMEL usage and it is good to test the installation and package completion before the large scale analysis. It currently
| has three files for unsupervised, supervised, and metric leagning using the simple swiss_roll data from sklearn.