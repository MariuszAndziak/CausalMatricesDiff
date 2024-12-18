# Causal Matrices Differences #
[![Generic badge](https://img.shields.io/badge/languge-english-blue.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/causal%20ai-8A2BE2)](https://www.shields.io/)

> This project provides a toolkit for analyzing and visualizing differences between causal matrices, enabling users to better understand discrepancies in predicted causal structures. By comparing true and predicted causal matrices, it highlights missing and additional causal relationships through intuitive visualizations.

The Structural Hamming Distance is calculated to quantify the overall difference between the true and predicted causal structures, offering a numeric measure of prediction accuracy. Additionally, the tool computes the number of undirected edges in both matrices to evaluate symmetry in causal relationships and better understand the directionality of the predictions.

A textual report is also generated, detailing false positive and false negative causal relationships, providing an easy-to-read summary of the mismatches for better interpretability and deeper insights into model performance.

## **Constructed Implementation** ## 
<p align="center">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/example.png" alt="Show differences in DAG structures">
</p>

## **Example of usage** ##
```python
from CausalMatricesDiff import CausalMatricesDiff as CMD

true_dag = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]
)

pred_dag = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]
)

names = ['Guinness', 'Whiskey', 'Katie', 'Cocoa', 'Harley', 'Scout', 'Chloe', 'Millie', 'Winnie', 'Wrigley']

compare_matrices = CMD(true_dag=true_dag, pred_dag=pred_dag, var_names=names)

compare_matrices.plot_causal_matrices_diff(show_legend=True)

metrics = compare_matrices.metrics()

print(compare_matrices.list_differences())
print('#'*30)
print('Structural Hamming Distance:' ,metrics['shd'])
print(metrics['undir'])
```

<p align="center">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/example2.png" alt="Show differences in DAG structures">
</p>

```
Pred DAG doesn't have causal paths from:
- Katie to Millie
which are present in True DAG
------------------------------
Pred DAG has additional causal paths from:
- Guinness to Millie
- Whiskey to Millie
- Katie to Cocoa
- Cocoa to Harley
- Cocoa to Scout
which are not present in True DAG
##############################
Structural Hamming Distance: 6.0
{'# of undirected edges for True DAG': 0, '# of undirected edges for Pred DAG': 0}
```