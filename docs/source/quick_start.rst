Quick Start
=============

The typical work flow using CAMEL is very similar to the sklearn and many other dimention reduction techniques, such as UMAP, Trimap, PacMAP. Acturally, the development of CAMEL is based on the above-mentioned packages.

The most common flow chart using CAMEL is 

- Load data (feature, labels(optional))
- Set embedding paramaters (optional and default values are set in the CAMEL).
- Perform defined operations in the CAMEL() class (fit_transform, transform, and inverse_transform in the current version)
- Evaluation using metrics (optional, CAMEL git repo provides 14 metrics to evaluate the global and local embedding quantities)
- Plotting the embedding results (optional, vilulize the results in low dimentional or high dimentional space)

Several templates are provided in this tutorial following the above-mentioned flow chart for unsupervised learning, supervised learning, semi_supervised learning, metric learning, and inverse learning.

In addtion, CAMEL can serve as feature extraction tool for your specific applications.

- Apply CAMEL to obatin low-dimentional feature representation (unsupervised or supervised)
- clustering and classification (with any classifier, e.g., NN-based or Optics)
- regression and prediction (with any nonlinear regressor such as Gaussian Process model)
- perform self_supervised_learning task using the reduced dimention feature (ongoing work and examples coming soon)

I am planning to releases a few citific and engineering application examples to illusrate the CAMEL useage (coming soon).
