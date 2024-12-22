import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import warnings
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from typing import List, Optional, Tuple, Dict, Union

warnings.filterwarnings("ignore")


class CausalMatricesDiff:
    """
    Visualize differences between two causal matrices.
    """

    def __init__(
        self, 
        true_dag: np.ndarray, 
        pred_dag: np.ndarray, 
        var_names: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the class with true and predicted DAG matrices and optional variable names.

        Args:
            true_dag (np.ndarray): The ground truth DAG matrix.
            pred_dag (np.ndarray): The predicted DAG matrix.
            var_names (Optional[List[str]]): List of variable names. Defaults to indices if not provided.
        """
        self.true_dag: np.ndarray = true_dag
        self.pred_dag: np.ndarray = pred_dag
        self.fn_list, self.fp_list = self.get_not_equal()
        self.var_names: List[str] = var_names or [
            f"variable_{n}" for n in range(self.true_dag.shape[0])
        ]

    def get_not_equal(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Identify false negatives and false positives by comparing the true and predicted DAGs.

        Returns:
            Tuple[List[List[int]], List[List[int]]]: Lists of false negatives and false positives.
        """
        result_fn: List[List[int]] = []
        result_fp: List[List[int]] = []

        for i in range(self.pred_dag.shape[0]):
            for j in range(self.pred_dag.shape[0]):
                if self.pred_dag[i, j] == 0 and self.true_dag[i, j] == 1:
                    result_fn.append([i, j])
                if self.pred_dag[i, j] == 1 and self.true_dag[i, j] == 0:
                    result_fp.append([i, j])

        return result_fn, result_fp

    def plot_causal_matrices_diff(
        self,
        save_name: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        cmap_name: str = "Greys",
        show_only_one_plot = True,
        show_legend = False
    ) -> None:
        """
        Visualize the differences between the true and predicted DAG matrices.

        Args:
            save_name (Optional[str]): If provided, saves the plot to the specified file path.
            figsize (Tuple[int, int]): Figure size for the plot.
            cmap_name (str): Colormap to use for the heatmap.
            show_only_one_plot (bool): Show only predicted DAG or also true DAG.
            show_legend (bool): Show square color codes.
        """
        assert self.true_dag.shape[0] == len(
            self.var_names
        ), "Length of variable list doesn't match the shape of the causal matrix"

        N = 1 if show_only_one_plot else 2

        fig, ax = plt.subplots(ncols=N, figsize=figsize)

        if N == 1:
            ax = [ax]

        plt.style.use('ggplot')

        for n in range(N):
            sns.heatmap(
                self.pred_dag if n == 0 else self.true_dag,
                cmap=cmap_name,
                ax=ax[n],
                cbar=False,
                linewidths=0,
                linecolor="k",
                facecolor="white",
                edgecolor="k",
            )
            ax[n].set_xticklabels(self.var_names, rotation=90)
            ax[n].set_yticklabels(self.var_names, rotation=0)
            ax[n].set_aspect("equal")

        ax[0].set_title("Pred DAG")

        if show_only_one_plot == False:
            ax[1].set_title("True DAG")

        for j, i in self.fp_list:
            ax[0].add_patch(
                Rectangle((i, j), 1, 1, linewidth=0, facecolor="#D3D3D3", edgecolor="k")
            )

        for j, i in self.fn_list:
            ax[0].add_patch(
                Rectangle((i, j), 1, 1, linewidth=2, facecolor="white", edgecolor="k")
            )
        
        legend_handles = [
    Line2D([0], [0], marker='s', color='black', markersize=10, label='Present in True DAG and Pred DAG', linestyle='None'),
    Line2D([0], [0], marker='s', markerfacecolor='white', markeredgecolor='black', markersize=10, label='Present only in True DAG', linestyle='None'),
    Line2D([0], [0], marker='s', color='#D3D3D3', markersize=10, label='Present only in Pred DAG', linestyle='None'),
]


        if save_name is not None:
            fig.savefig(save_name)
        else:
            plt.tight_layout()
            
            if show_legend:
                plt.legend(
                    handles = legend_handles,
                    loc='upper left',
                    frameon=True
                )
            plt.show()

    def number_of_undirected(self) -> Dict[str, float]:
        """
        Count the number of undirected edges in the true and predicted DAGs.

        Returns:
            Dict[str, float]: Counts of undirected edges for both DAGs.
        """
        total_true: float = 0
        total_pred: float = 0

        for i in range(self.true_dag.shape[0]):
            for j in range(self.true_dag.shape[0]):
                if self.true_dag[i, j] == 1 and self.true_dag[i, j] == self.true_dag[j, i]:
                    total_true += 0.5
                if self.pred_dag[i, j] == 1 and self.pred_dag[i, j] == self.pred_dag[j, i]:
                    total_pred += 0.5

        return {
            "# of undirected edges for True DAG": total_true,
            "# of undirected edges for Pred DAG": total_pred,
        }

    def structural_hamming_distance(self) -> float:
        """
        Compute the Structural Hamming Distance (SDH) between the True DAG and Pred DAG.

        Returns:
            float: The sum of absolute differences between the two DAGs.
        """
        diff = np.abs(self.true_dag - self.pred_dag)
        return float(np.sum(diff))
    
    def metrics(self) -> dict:
        """
        Bind metrics into one dictionary.

        Returns:
            dict: A dictionaty with all the metrics for easy access.

        """
        metrics = dict()
        metrics['shd'] = self.structural_hamming_distance()
        metrics['undir'] = self.number_of_undirected()
        return metrics
    
    def format_differences_report(self,
        false_negatives: List[List[str]] = None, 
        false_positives: List[List[str]] = None
    ) -> str:
        """
        Generate a formatted report of differences between true and predicted DAGs.

        Args:
            false_negatives (List[List[str]]): List of false negative variable pairs.
            false_positives (List[List[str]]): List of false positive variable pairs.

        Returns:
            str: A formatted string describing the differences.
        """
        report = []

        if false_negatives == None and false_positives == None:
            false_positives, false_negatives = self.get_not_equal()

        report.append("Pred DAG doesn't have causal paths from:")
        for elem in false_negatives:
            report.append(f"- {elem[0]} to {elem[1]}")
        report.append("which are present in True DAG")
        report.append("-" * 30)
        report.append("Pred DAG has additional causal paths from:")
        for elem in false_positives:
            report.append(f"- {elem[0]} to {elem[1]}")
        report.append("which are not present in True DAG")

        return "\n".join(report)


    def list_differences(
        self, return_text_description: bool = True
    ) -> Union[None, Dict[str, List[List[str]]], str]:
        """
        List the differences between the true and predicted DAGs in terms of false positives
        and false negatives.

        Args:
            return_text_description (bool): Whether to print a text description or return a dictionary.

        Returns:
            Union[None, Dict[str, List[List[str]]], str]: Formatted string if return_text_description is True,
            or a dictionary of false negatives and positives otherwise.
        """
        false_negatives_variables = [
            [self.var_names[x], self.var_names[y]] for x, y in self.fn_list
        ]
        false_positives_variables = [
            [self.var_names[x], self.var_names[y]] for x, y in self.fp_list
        ]

        if return_text_description:
            return self.format_differences_report(
                false_negatives=false_negatives_variables, 
                false_positives=false_positives_variables
            )
        else:
            return {
                "False Negatives": false_negatives_variables,
                "False Positives": false_positives_variables,
            }
    
    def draw_dag(self, layout_num: int = -1) -> None:
        """
        Draws the Directed Acyclic Graph (DAG) with options for different layouts.
        
        Highlights false positives with red edges and uses black edges for all other connections.
        Allows the user to select a specific layout or visualize the graph using all available layouts.

        Args:
            layout_num (int): Specifies the layout to use. If set to -1, all layouts will be drawn sequentially.

            Available Layouts:
            0: Spring Layout (Force-directed)
            1: Circular Layout
            2: Kamada-Kawai Layout
            3: Shell Layout
            4: Spectral Layout
            5: Planar Layout
            6: Spiral Layout
            7: Random Layout
        """
        # Define available layouts
        layouts: Dict[int, callable] = {
            0: nx.spring_layout,     # Force-directed layout
            1: nx.circular_layout,   # Nodes arranged in a circle
            2: nx.kamada_kawai_layout,  # Optimized force-directed layout
            3: nx.shell_layout,      # Nodes in concentric shells
            4: nx.spectral_layout,   # Based on graph Laplacian
            5: nx.planar_layout,     # Non-overlapping edges (if planar)
            6: nx.spiral_layout,     # Nodes arranged in a spiral
            7: nx.random_layout      # Random node positions
        }

        # Get false positives and negatives from the DAG comparison
        false_positives = self.list_differences(return_text_description=False)['False Positives']
        false_negatives = self.list_differences(return_text_description=False)['False Negatives']

        # Create a directed graph
        G: nx.DiGraph = nx.DiGraph()
        for i, row in enumerate(self.pred_dag):
            for j, value in enumerate(row):
                if value == 1:
                    G.add_edge(self.var_names[i], self.var_names[j])  # Add edge based on prediction DAG

        # Highlight false positives with red edges
        red_edges: List[Tuple[str, str]] = []
        for u, v in false_positives:
            G.add_edge(u, v)  # Add edge for false positives
            red_edges.append((u, v))

        # Black edges for all other connections
        black_edges = [edge for edge in G.edges if edge not in red_edges]

        # If a specific layout is chosen, filter to that layout only
        if layout_num != -1:
            layouts = {layout_num: layouts[layout_num]}

        # Draw the graph for each selected layout
        for layout_number, layout in layouts.items():
            pos = layout(G)  # Compute node positions

            # Draw the graph
            nx.draw(
                G, pos, with_labels=True, node_size=2000, font_size=10, font_color='white'
            )
            # Draw black edges
            nx.draw_networkx_edges(G, pos, edgelist=black_edges, edge_color='black')
            # Draw red edges with increased width
            nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', width=3)

            # Add a title for the layout
            if layout_num == -1:
                plt.title(f"[{layout_number}] {layout.__name__}")
            plt.show()
    
    def legend_descriptions(self, show = 'both'):
        descriptions = [
            '''
            Red edges are false positives - a path present in Pred DAG but absent in True DAG. \n
            Grey edges are false negatives - a path present in True DAG but absent in Pred DAG. \n
            Black edges present matching paths in True DAG and Pred DAG.
            ''',

            '''
            White squares represent a connections from variable in on the X axis to variable on Y axis only in True DAG.
            Grey squares represent a connections from variable in on the X axis to variable on Y axis only in Pred DAG.
            Black squares present a match in True DAG and Pred DAG.
            '''
        ]
        if show == 'dag':
            return descriptions[0]
        elif show == 'matrix':
            return descriptions[1]
        else:
            return '\n'.join(descriptions)
    
    def calculate_match_percentage(self) -> dict:
        """
        Calculates the percentage of edges in the predicted DAG (pred_dag) that match the edges in 
        the true DAG (true_dag) and the number of additional edges that are present in pred_dag,

        Args:
            true_dag (np.ndarray): The ground truth DAG as a binary adjacency matrix (n x n).
            pred_dag (np.ndarray): The predicted DAG as a binary adjacency matrix (n x n).

        Returns:
            dict: The match percentage (0-100) indicating the proportion of matched edges. 
            The total number of false positives ini pred_dag.
        """
        if self.true_dag.shape != self.pred_dag.shape:
            raise ValueError("The true_dag and pred_dag must have the same dimensions.")

        # Total edges in the true DAG
        total_true_edges = np.sum(self.true_dag)

        # Matched edges between true DAG and predicted DAG
        matched_edges = np.sum((self.true_dag == 1) & (self.pred_dag == 1))

        # Avoid division by zero if the true DAG has no edges
        if total_true_edges == 0:
            return 100.0 if np.sum(self.pred_dag) == 0 else 0.0

        # Calculate the match percentage
        match_percentage = 100 * matched_edges / total_true_edges
        
        false_positives_len = len(self.list_differences(return_text_description=False)['False Positives'])
        return {
            'matched paths': match_percentage,
            'additional paths': false_positives_len
            }




