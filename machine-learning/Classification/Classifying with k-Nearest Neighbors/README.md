The k-Nearest Neighbors algorithm is a simple and effective way to classify data. kNN is
an example of instance-based learning, where you need to have instances of data close
at hand to perform the machine learning algorithm. _The algorithm has to carry
around the full dataset;_ for large datasets, this implies a large amount of storage. In
addition, you need to calculate the distance measurement for every piece of data in
the database, and this can be cumbersome.

_An additional drawback is that kNN doesn’t give you any idea of the underlying
structure of the data;_ you have no idea what an “average” or “exemplar” instance from
each class looks like. In the next chapter, we’ll address this issue by exploring ways in
which probability measurements can help you do classification.