import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

class CausalMatricesDiff(object):
    '''
    Visualize differences between two causal matrices. 
    '''
    def __init__(self, true_dag, pred_dag):
        self.true_dag = true_dag
        self.pred_dag = pred_dag
    

    def causal_matrices_diff(self, var_names = None, save_name = None, figsize=(12,6), cmap_name='Greys'):

        def get_not_equal(self):
            result_fn = []; result_fp = []
            for i in range(self.pred_dag.shape[0]):
                for j in range(self.pred_dag.shape[0]):
                    if (self.pred_dag[i,j] == 0) and (self.true_dag[i,j] == 1):
                        result_fn.append([i,j])
                    if (self.pred_dag[i,j] == 1) and (self.true_dag[i,j] == 0):
                        result_fp.append([i,j])
            return result_fn, result_fp
        
        fn_list, fp_list = get_not_equal(self)
        
        if var_names == None:
            var_names = list(range(self.true_dag.shape[0]))

        assert self.true_dag.shape[0] == len(var_names), "Length of variable list doesn't match the shape of the causal matrix"

        fig, ax = plt.subplots(ncols = 2, figsize=figsize)
        for n in range(2):

            sns.heatmap(self.true_dag, cmap=cmap_name, ax = ax[n], cbar=False, linewidths=0, linecolor='k', facecolor='white', edgecolor='k');
            ax[n].set_xticklabels(var_names, rotation=90);
            ax[n].set_yticklabels(var_names, rotation=0);
            ax[n].set_aspect('equal');

        ax[0].set_title('True DAG')
        ax[1].set_title('Pred DAG')

        for j,i in fp_list:
            ax[1].add_patch(Rectangle((i, j), 1, 1, linewidth=0, facecolor='#D3D3D3', edgecolor='k'))

        for j,i in fn_list:
            ax[1].add_patch(Rectangle((i, j), 1, 1, linewidth=2, facecolor = 'white', edgecolor='k'))
        
        if save_name is not None:
            fig.savefig(save_name)
        else:
            plt.tight_layout()

    def number_of_undirected(self):
        total_true = 0
        total_pred = 0
        for i in range(self.true_dag.shape[0]):
            for j in range(self.true_dag.shape[0]):
                if (self.true_dag[i,j] == 1) and (self.true_dag[i,j] == self.true_dag[j,i]):
                    total_true +=0.5
                if (self.pred_dag[i,j] == 1) and (self.pred_dag[i,j] == self.pred_dag[j,i]):
                    total_pred +=0.5
        return {
            '# of undirected edges for true_dag': total_true, 
            '# of undirected edges for pred_dag': total_pred
            }

    def structural_humming_distance(self):
        diff = abs(self.true_dag - self.pred_dag)
        return np.sum(diff)

true_dag = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 1., 0., 1.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 0., 1., 1., 0., 0., 1., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 1., 0., 0., 0., 1., 0.]])

pred_dag = np.array([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 1., 0., 1., 0., 0., 0., 0., 0., 1.],
       [0., 1., 0., 0., 1., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 0., 1., 1., 0., 0., 1., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 1., 0., 0., 0., 1., 0.]])

var_names = [f'variable_{n}' for n in range(10)]
matrices = CausalMatricesDiff(true_dag, pred_dag)
matrices.causal_matrices_diff(save_name='example.png', var_names=var_names)
print(matrices.number_of_undirected())
print(matrices.structural_humming_distance())