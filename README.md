# Causal Matrices Differences #
[![Generic badge](https://img.shields.io/badge/languge-english-blue.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/causal%20ai-8A2BE2)](https://www.shields.io/)

> This project provides a toolkit for analyzing and visualizing differences between causal matrices, enabling users to better understand discrepancies in predicted causal structures. By comparing true and predicted causal matrices, it highlights missing and additional causal relationships through intuitive visualizations.

The Structural Hamming Distance is calculated to quantify the overall difference between the true and predicted causal structures, offering a numeric measure of prediction accuracy. Additionally, the tool computes the number of undirected edges in both matrices to evaluate symmetry in causal relationships and better understand the directionality of the predictions.

A textual report is also generated, detailing false positive and false negative causal relationships, providing an easy-to-read summary of the mismatches for better interpretability and deeper insights into model performance.

## **Constructed Implementation** ## 
<p align="center">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/graphics/example.png" alt="Show differences in DAG structures">
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

compare = CCM(true_dag=true_dag, pred_dag=pred_dag, var_names = names)
compare.standard_pipeline(print_step_names=True)
```


<p align="center">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/graphics/example2.png" alt="Show differences in DAG structures">
</p>


```bash
##### Performing 'format_differences_report' ##### 
True DAG has unique causal paths from:
- 2 to 7
which are not present in Pred DAG
------------------------------
Pred DAG has unique causal paths from:
- 0 to 7
- 1 to 7
- 2 to 3
- 3 to 4
- 3 to 5
which are not present in True DAG

##### Performing 'list_differences' ##### 
True DAG has unique causal paths from:
- Katie to Millie
which are not present in Pred DAG
------------------------------
Pred DAG has unique causal paths from:
- Guinness to Millie
- Whiskey to Millie
- Katie to Cocoa
- Cocoa to Harley
- Cocoa to Scout
which are not present in True DAG

##### Performing 'metrics' ##### 
{'shd': 6.0, 'undir': {'# of undirected edges for True DAG': 0, '# of undirected edges for Pred DAG': 0}}
##### Performing 'draw_dags' ##### 
```
<p align="center">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/graphics/d0.png">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/graphics/d1.png">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/graphics/d2.png">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/graphics/d3.png">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/graphics/d4.png">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/graphics/d5.png">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/graphics/d6.png">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/graphics/d7.png">
</p>

```bash
##### Performing 'legend_description' ##### 

            Red edges are false positives - a path present in Pred DAG but absent in True DAG. 

            Grey edges are false negatives - a path present in True DAG but absent in Pred DAG. 

            Black edges present matching paths in True DAG and Pred DAG.
            

            White squares represent a connections from variable in on the X axis to variable on Y axis only in True DAG.
            Grey squares represent a connections from variable in on the X axis to variable on Y axis only in Pred DAG.
            Black squares present a match in True DAG and Pred DAG.
            

##### Performing 'calculate_match_percentage' ##### print_step_names=True
```
