# need to install the following packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



global _RANDOM_STATE
_RANDOM_STATE = None


output_path = "../output/2/"

# methods_compare= ['CAMEL']
# data_compare = ['MNIST']



#load all results in output folder        
total_time=np.load(output_path + '/total_time.npy')
metrics_knn=np.load(output_path + '/metrics_knn.npy')
metrics_svm=np.load(output_path + '/metrics_svm.npy')
metrics_triplet=np.load(output_path + '/metrics_triplet.npy')
metrics_nkr=np.load(output_path + '/metrics_nkr.npy')
metrics_scorr=np.load(output_path + '/metrics_scorr.npy')
metrics_cenknn=np.load(output_path + '/metrics_cenknn.npy')
metrics_cencorr=np.load(output_path + '/metrics_cencorr.npy')
metrics_clusterratio=np.load(output_path + '/metrics_clusterratio.npy')
metrics_coranking_auc=np.load(output_path + '/metrics_coranking_auc.npy')
metrics_coranking_trust=np.load(output_path + '/metrics_coranking_trust.npy')
metrics_coranking_cont=np.load(output_path + '/metrics_coranking_cont.npy')
metrics_coranking_lcmc=np.load(output_path + '/metrics_coranking_lcmc.npy')
metrics_curvature_simi=np.load(output_path + '/metrics_curvature_simi.npy')
metrics_nnwr=np.load(output_path + '/metrics_nnwr.npy')
methods_compare=np.load(output_path + '/methods_compare.npy')
data_compare=np.load(output_path + '/data_compare.npy')


#compute mean and std from multiple runs
n_monte=total_time.shape[0]
n_data=total_time.shape[1]
n_method=total_time.shape[2]

avg_time=np.mean(total_time,axis=0)
std_time=np.std(total_time,axis=0)
avg_time_speed_max=np.divide(np.max(avg_time[:,:n_method-1],axis=1),avg_time[:,n_method-1])
avg_time_speed_min=np.divide(np.min(avg_time[:,:n_method-1],axis=1),avg_time[:,n_method-1])
avg_knn=np.mean(metrics_knn, axis=0)
std_knn=np.std(metrics_knn, axis=0)
avg_svm=np.mean(metrics_svm, axis=0)
std_svm=np.std(metrics_svm, axis=0)
avg_triplet=np.mean(metrics_triplet, axis=0)
std_triplet=np.std(metrics_triplet, axis=0)
avg_nkr=np.mean(metrics_nkr, axis=0)
std_nkr=np.std(metrics_nkr, axis=0)
avg_scorr=np.mean(metrics_scorr, axis=0)
std_scorr=np.std(metrics_scorr, axis=0)
avg_cenknn=np.mean(metrics_cenknn, axis=0)
std_cenknn=np.std(metrics_cenknn, axis=0)
avg_cencorr=np.mean(metrics_cencorr, axis=0)
std_cencorr=np.std(metrics_cencorr, axis=0)
avg_clusterratio=np.mean(metrics_clusterratio, axis=0)
std_clusterratio=np.std(metrics_clusterratio, axis=0)
avg_coranking_auc=np.mean(metrics_coranking_auc, axis=0)
std_coranking_auc=np.std(metrics_coranking_auc, axis=0)
avg_coranking_trust=np.mean(metrics_coranking_trust, axis=0)
std_coranking_trust=np.std(metrics_coranking_trust, axis=0)
avg_coranking_cont=np.mean(metrics_coranking_cont, axis=0)
std_coranking_cont=np.std(metrics_coranking_cont, axis=0)
avg_coranking_lcmc=np.mean(metrics_coranking_lcmc, axis=0)
std_coranking_lcmc=np.std(metrics_coranking_lcmc, axis=0)
avg_curvature_simi=np.mean(metrics_curvature_simi, axis=0)
std_curvature_simi=np.std(metrics_curvature_simi, axis=0)
avg_nnwr=np.mean(metrics_nnwr, axis=0)
std_nnwr=np.std(metrics_nnwr, axis=0)


# Set up the grid
fig_rows=5
fig_colums=3
fig = plt.figure(figsize=(15*fig_colums,10*fig_rows),layout='constrained',dpi=300)
gs = GridSpec(fig_rows, fig_colums, figure=fig)

digit_axes = np.zeros((fig_rows, fig_colums), dtype=object)


metrics_list=[['kNN', 'SVM', 'Triplet'],
            ['NPP', 'Spear-Corr','Cen-kNN'],
            ['Cen-Dist','Trust', 'Conti',],
            ['LCMC', 'AUC','Cluster-Ratio'],
            ['Curvature-Simi','NNWR', 'Empty']]


for i in range(fig_rows):
    for j in range(fig_colums):
        
        if metrics_list[i][j] == 'kNN':
            avg_plot=avg_knn
            std_plot=std_knn
        elif metrics_list[i][j] == 'SVM':
            avg_plot=avg_svm
            std_plot=std_svm
        elif metrics_list[i][j] == 'Triplet':
            avg_plot=avg_triplet
            std_plot=std_triplet
        elif metrics_list[i][j] == 'NPP':
            avg_plot=avg_nkr
            std_plot=std_nkr            
        elif metrics_list[i][j] == 'Spear-Corr':
            avg_plot=avg_scorr
            std_plot=std_scorr
        elif metrics_list[i][j] == 'Cen-kNN':
            avg_plot=avg_cenknn
            std_plot=std_cenknn
        elif metrics_list[i][j] == 'Cen-Dist':
            avg_plot=avg_cencorr
            std_plot=std_cencorr
        elif metrics_list[i][j] == 'Trust':
            avg_plot=avg_coranking_trust
            std_plot=std_coranking_trust
        elif metrics_list[i][j] == 'Conti':
            avg_plot=avg_coranking_cont
            std_plot=std_coranking_cont
        elif metrics_list[i][j] == 'LCMC':
            avg_plot=avg_coranking_lcmc
            std_plot=std_coranking_lcmc
        elif metrics_list[i][j] == 'AUC':
            avg_plot=avg_coranking_auc
            std_plot=std_coranking_auc
        elif metrics_list[i][j] == 'Cluster-Ratio':
            avg_plot=avg_clusterratio
            std_plot=std_clusterratio
        elif metrics_list[i][j] == 'Curvature-Simi':
            avg_plot=avg_curvature_simi
            std_plot=std_curvature_simi
        elif metrics_list[i][j] == 'NNWR':
            avg_plot=avg_nnwr
            std_plot=std_nnwr
        elif metrics_list[i][j] == 'Empty':
            avg_plot=0.0
            std_plot=0.0           
        else:
            print('Unsupported metrics')
            assert(False)




        if metrics_list[i][j] != 'Empty':
            digit_axes[i, j] = fig.add_subplot(gs[i, j])
            x = np.arange(len(data_compare))  # the label locations
            width = 0.15  # the width of the bars
            multiplier = 0
            for ii in range(avg_knn.shape[1]):
                offset = width * multiplier
                rects = digit_axes[i, j].bar(x + offset, avg_plot[:,ii], width, yerr=std_plot[:,ii], label=methods_compare[ii])
                # ax.bar_label(rects, padding=3)
                multiplier += 1
    
            # Add some text for labels, title and custom x-axis tick labels, etc.
            digit_axes[i, j].set_ylabel(metrics_list[i][j]+' score', fontsize=20)
            digit_axes[i, j].set_title('Model Comparison using ' + metrics_list[i][j],fontsize=40)
            digit_axes[i, j].set_xticks(x + width, data_compare)
            digit_axes[i, j].set_xticklabels(data_compare, fontsize=20, rotation = 45)
            digit_axes[i, j].legend(loc='upper left', fontsize=20, ncols=3)
            digit_axes[i, j].set_ylim(0, 1.5)

plt.show()


x = np.arange(len(data_compare))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', dpi=300)

for i in range(avg_knn.shape[1]):
    offset = width * multiplier
    rects = ax.bar(x + offset, avg_time[:,i], width, yerr=std_time[:,i],label=methods_compare[i])
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Wall Clock Time (s)', fontsize=14)
ax.set_title('Model Comparison for Computing Time',fontsize=16)
ax.set_xticks(x + width, data_compare)
ax.set_xticklabels(data_compare, fontsize=14, rotation = 45)
ax.legend(loc='upper left', ncols=3, fontsize=12)
ax.set_ylim(0, 90)

plt.show()

