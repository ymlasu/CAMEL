Semisupervised Learning
======================

Semisupervised learning is between the unsupervised and supervised learning of CAMEL. A good semisupervised learning algorithm should be able to 
* requir miminimum amount of labeld data to represent the fully labeld results
* reproduce unsupervised learning when the available labels are near zero and supervised learning when the available labels are almost the full.

We will use several examples below with code to illustrate how to use CAMEL to perform the semisupervised learning task.



Comprehensive example
-----------------------

A comprehensive exmaple is shown here, which is the template (semi_supervised_learning_compare.py) under the folder demo in the git repo. Dpemneding on your stored location, the path in the code might need to be updated.

The first step is to import all nessary modulues and load data function is the same as the unsupervised learning and is not repeated here.


.. code:: python

    data_path = "../data/"
    output_path = "../output/semi_supervised_learning/"
    methods_compare= ['UMAP','CAMEL']
    data_compare = ['swiss_roll', 'MNIST']
    weight_list = np.array([0.0, 1e-4, 0.2, 0.9, 0.99])

The above code specifies the data_path and output_path. If you downloaded the fit and directly work on the files in the /demo folder, then you do not need to change these paths as they are referring to the \data folder and \output folder in the parent directory.

The above code also specifies the methods checked and material database used. It also checks the effect of different label ratio (from 0 to 1.0) for the semisupervised learning. Since only UMAP and CAMEL provide this functionality, only these two methods are checked.


.. code:: python

    n_monte=5

    n_methods=len(methods_compare)
    n_data=len(data_compare)
    n_weight=len(weight_list)

    metrics_knn=np.zeros([n_monte,n_data,n_methods])
    metrics_svm=np.zeros([n_monte,n_data,n_methods])
    metrics_triplet=np.zeros([n_monte,n_data,n_methods])
    metrics_nkr=np.zeros([n_monte,n_data,n_methods])
    metrics_scorr=np.zeros([n_monte,n_data,n_methods])
    metrics_cenknn=np.zeros([n_monte,n_data,n_methods])
    metrics_cencorr=np.zeros([n_monte,n_data,n_methods])
    metrics_clusterratio=np.zeros([n_monte,n_data,n_methods])
    metrics_coranking_auc=np.zeros([n_monte,n_data,n_methods])
    metrics_coranking_trust=np.zeros([n_monte,n_data,n_methods])
    metrics_coranking_cont=np.zeros([n_monte,n_data,n_methods])
    metrics_coranking_lcmc=np.zeros([n_monte,n_data,n_methods])
    metrics_curvature_simi=np.zeros([n_monte,n_data,n_methods])
    metrics_nnwr=np.zeros([n_monte,n_data,n_methods])

Since the embedding is random, the performance check may need multiple runs and n_monte is the number of Monte Carlo simulations. The code also zeros many matrices to store the metrics computiing.
metrics_XXXX referes to the computing of XXXX (name of metrics and can be found in the arXiv paper for details).

.. code:: python

    # Set up the grid
    fig = plt.figure(figsize=(8*n_weight,6*n_data*n_methods),layout='constrained',dpi=300)
    gs = GridSpec(n_data*n_methods, n_weight, figure=fig)

    digit_axes = np.zeros((n_data*n_methods, n_weight), dtype=object)

Since there are several methods and datasets, the visulization is orgnized using grid matlibplot. You can use this as the template for other type of grid plot.

.. code:: python

    for k in range(n_methods):
        
                ........
                elif methods_compare[k]  == 'UMAP':
                    transformer = umap.UMAP(target_metric=target_metric, random_state=1)
                    y_semi=np.copy(y[:10000])

                    y_semi[int(weight_list[j]*10000):]=-1
                    
                    if weight_list[j] < 1e-8:
                        X_embedding = transformer.fit_transform(X)
                    elif weight_list[j] > 0.9999:
                        X_embedding = transformer.fit_transform(X, y)                    
                    else:
                        X_embedding = transformer.fit_transform(X, y_semi)
    
                elif methods_compare[k] == 'TSNE':
                    transformer = TSNE()
                elif methods_compare[k]  == 'TriMAP':
                    transformer = trimap.TRIMAP()
                elif methods_compare[k]  == 'CAMEL':
                    transformer = CAMEL(n_neighbors=10, FP_number=20, w_neighbors=1.0, 
                                        tail_coe=0.05, w_curv=0.001, w_FP=20, num_iters=400, target_type=target_type, random_state=1)     
                    y_semi=np.copy(y[:10000])

                    if y_semi is not None and isinstance (y_semi, (np.ndarray)):
                        y_semi=pd.DataFrame(data=y_semi,columns=['labels'])

                    y_semi.loc[int(weight_list[j]*10000):,'labels']=pd.NA
                    
                    if weight_list[j] < 1e-8:
                        X_embedding = transformer.fit_transform(X)
                    elif weight_list[j] > 0.9999:
                        X_embedding = transformer.fit_transform(X, y)                    
                    else:
                        X_embedding = transformer.fit_transform(X, y_semi)

                else:
                    print("Incorrect method specified")
                    assert(False)
            


                y_plot = np.copy(y).astype(int)
        
                # Visualization
                

                
                digit_axes[k*n_data+i, j] = fig.add_subplot(gs[k*n_data+i, j])
                digit_axes[k*n_data+i, j].scatter(X_embedding[:, 0], X_embedding[:, 1],
                                    c=y_plot, cmap='jet', s=0.2)
                title_embedding = 'missing ratio of '+ str(1-weight_list[j])
                digit_axes[k*n_data+i, j].set_title(title_embedding,fontsize=12)
                digit_axes[k*n_data+i, j].set_axis_off()


The above code shows how to construct the semi_supervised data set. In UMAP, the unknown labels are marked as "-1". In CAMEL, the unknown labels are marked as "NaN". This is achvied using pd.NA function. Thus, missing label data can be automatically handled in realistic data.

Finally, all results are saved in the specified output path. Once all done, you can check the visulization of embedding results.

The left most column is the unsupervised learning and the right most column is the fully supervised learning. From left to the right, the ratio of available labels increases. As can be seen, compared to UMAP, CAMEL appears to have a stable and smooth transition.

.. image:: ../semi_supervised_model_compare.png
  :width: 600
  :alt: supervised_model_compare
  :align: center