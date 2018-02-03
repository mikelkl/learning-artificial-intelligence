A decision tree classifier is just like a work-flow diagram with the terminating blocks
representing classification decisions. Starting with a dataset, you can measure the
inconsistency of a set or the entropy to find a way to split the set until all the data
belongs to the same class. The ID3 algorithm can split nominal-valued datasets. Recursion
is used in tree-building algorithms to turn a dataset into a decision tree. The tree
is easily represented in a Python dictionary rather than a special data structure.

Cleverly applying Matplotlib’s annotations, you can turn our tree data into an easily
understood chart. The Python Pickle module can be used for persisting our tree.
The contact lens data showed that decision trees can try too hard and overfit a dataset.
This overfitting can be removed by pruning the decision tree, combining adjacent
leaf nodes that don’t provide a large amount of information gain.

[Download data](https://drive.google.com/open?id=1e0KEUi1Y-fYl_FQvGdS8TM9aMC0NMYxI) into the same folder.