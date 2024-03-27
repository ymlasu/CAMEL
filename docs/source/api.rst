API
===

CAMEL class
------------

The camel package is inspired and developed based on many dimension reduction packages, such as UMAP, TriMAP, and PaCMAP, which follow a similar setting from sklearn classes. Thus, CAMEL shares a similar calling format using the CAMEL API.

1. There is only one class, CAMEL().
2. fit(X, y) and fit_transform(X, y) perform training in embedding data and constructing a "model". X refers to input feature data, and y refers to input label data. y is optional and can also have missing/NaN data. This module is mainly used for unsupervised, supervised, and semi-supervised learning.
3. transform(Xnew, basis) is for embedding if new testing data Xnew is provided and the model is constructed using basis datasets. Basis data is optional. This module is mainly used for metric learning, where the metric model is already learned from training data, whether it is supervised, unsupervised, or semi-supervised learning. 
4. invser_transform(ynews, X, y) is used for inverse embedding and dimension augmentation from low to high dimensions. This module assumes that you have a forward embedding constructed from training data X (basis feature) and y (embedding of basis feature). Then, one can reverse this process by constructing a feature space vector from a new unseen point in a dimension point. This is in analogy to the generative model from a latent space in ML. 


parameters
-----------

Several parameters can control the CAMEL's results and performance. Default values have been set if you want to start quickly. Below is a description of several main factors if you want to fine-tune the CAMEL.

- ''n_components'': int, default=2
        Dimensions of the embedded space. Typical values are 2 or 3. It can be any integer.

- ' ' n_neighbors'': int, default=10
        Number of neighbors considered for nearest neighbor pairs for local structure preservation.

- ''FP_number'': float, default=20
        Number of further points(e.g., 20 Further pairs per node)
        Further pairs are used for both local and global structure preservation.

- ''tail_coe'': float, default=0.05
        The parameter to control the attractive force of neighbors (1/(1+tail_coe*dij)**2), smaller values indicate flat tail, and it is not recommended to change.
    
- ''w_neighbors'': float, default=1.0
        weight coefficient for the attractive force of neighbors, large values indicate strong force for the same distance metric
        
- ''w_curv'': float, default=0.001
        weight coefficient for attractive/repulsive force due to local curvature, large values indicate strong force for the same distance metric        

- ''w_FP'': float, default=20
        weight coefficient for the repulsive force of far points, large values indicate strong force for the same distance metric    
    
- ''lr'': float, default=1.0
        The learning rate of the Adam optimizer for embedding. do not recommend changing.

- ''num_iters'': int, default=400
        The number of iterations for optimizing embedding. It is observed that 200 is sufficient for most cases, and 400 is used here for safety reasons.

- ''target_weight'': float, default=0.5
        weight factor for target/label during the supervised learning, 0 indicates no weight, and it reduces to unsupervised one,
        1 indicates infinity weight (set as a large value in practice.

- ''random_state'': int, optional
        Random state for the camel instance.
        Setting a random state is useful for repeatability.



The other setting can be seen in the source code and will be updated in future documentation.
