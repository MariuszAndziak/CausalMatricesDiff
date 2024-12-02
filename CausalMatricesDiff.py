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
    def __init__(self, true_dag, pred_dag, var_names = None):
        self.true_dag = true_dag
        self.pred_dag = pred_dag
        self.fn_list, self.fp_list = self.get_not_equal()
        self.var_names = var_names

        if self.var_names == None:
            self.var_names = list(range(self.true_dag.shape[0]))

    def get_not_equal(self):
        result_fn = []; result_fp = []
        for i in range(self.pred_dag.shape[0]):
            for j in range(self.pred_dag.shape[0]):
                if (self.pred_dag[i,j] == 0) and (self.true_dag[i,j] == 1):
                    result_fn.append([i,j])
                if (self.pred_dag[i,j] == 1) and (self.true_dag[i,j] == 0):
                    result_fp.append([i,j])
        return result_fn, result_fp

    def causal_matrices_diff(self, save_name = None, figsize=(12,6), cmap_name='Greys'):

        # if var_names == None:
        #     var_names = list(range(self.true_dag.shape[0]))

        assert self.true_dag.shape[0] == len(var_names), "Length of variable list doesn't match the shape of the causal matrix"

        fig, ax = plt.subplots(ncols = 2, figsize=figsize)
        for n in range(2):

            sns.heatmap(self.true_dag, cmap=cmap_name, ax = ax[n], cbar=False, linewidths=0, linecolor='k', facecolor='white', edgecolor='k');
            ax[n].set_xticklabels(self.var_names, rotation=90);
            ax[n].set_yticklabels(self.var_names, rotation=0);
            ax[n].set_aspect('equal');

        ax[0].set_title('True DAG')
        ax[1].set_title('Pred DAG')

        for j,i in self.fp_list:
            ax[1].add_patch(Rectangle((i, j), 1, 1, linewidth=0, facecolor='#D3D3D3', edgecolor='k'))

        for j,i in self.fn_list:
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
    
    def explain_differences(self, return_text_description = True):
        false_negatives_variables  = [[self.var_names[x], self.var_names[y]] for x, y in self.fn_list]
        false_positives_variables  = [[self.var_names[x], self.var_names[y]] for x, y in self.fp_list]

        if return_text_description:
            print("Pred DAG doesn't have causal paths from:")
            for elem in false_negatives_variables:
                print(f'- {elem[0]} to {elem[1]}')
            print('which are present in True DAG')
            print('-'*30)
            print("Pred DAG has addiational causal paths from:")
            for elem in false_positives_variables:
                print(f'- {elem[0]} to {elem[1]}')
            print('which are not present in True DAG')
        else:
            return {
                'False Negatives': false_negatives_variables,
                'False Positives': false_positives_variables
            }
        

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
matrices = CausalMatricesDiff(true_dag, pred_dag, var_names=var_names)
matrices.causal_matrices_diff(save_name='example.png')
print(matrices.number_of_undirected())
print(matrices.structural_humming_distance())
print(matrices.explain_differences(return_text_description=True))