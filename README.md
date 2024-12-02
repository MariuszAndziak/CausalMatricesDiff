# Causal Matrices Differences #
[![Generic badge](https://img.shields.io/badge/languge-english-blue.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/causal%20ai-8A2BE2)](https://www.shields.io/)

> Description

## **Constructed Implementation** ## 
<p align="center">
  <img src="https://github.com/MariuszAndziak/CausalMatricesDiff/blob/main/example.png" alt="Show differences in DAG structures">
</p>
```
* Number of undirected edges: {'# of undirected edges for true_dag': 0, '# of undirected edges for pred_dag': 0}
* Structural Humming Distance: 6.0
* Listing of differences:
Pred DAG doesn't have causal paths from:
- variable_2 to variable_7
which are present in True DAG
------------------------------
Pred DAG has additional causal paths from:
- variable_1 to variable_7
- variable_2 to variable_3
- variable_3 to variable_4
- variable_3 to variable_5
which are not present in True DAG
```

## **Example of usage** ##
```python
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

var_names = [f"variable_{n}" for n in range(10)]
matrices = CausalMatricesDiff(true_dag, pred_dag, var_names=var_names)
matrices.causal_matrices_diff(save_name="example.png")
print("* Number of undirected edges:", matrices.number_of_undirected())
print("* Structural Humming Distance:", matrices.structural_humming_distance())
print("* Listing of differences:")
print(matrices.list_differences(return_text_description=True))
```
