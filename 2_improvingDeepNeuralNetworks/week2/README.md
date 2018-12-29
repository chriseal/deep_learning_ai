# Optimization Algorithms

## Mini-batch gradient descent

- let gradient descent make some progress before iterating through entire training set
- helps speed things up with really large datasets
- say, m=5M, minibatch has 1000 m in each --> 5K minibatchs
- X<sup>{1-5000}</sup>... {} denotes minibatch index
- 1 epoch is a single pass though the training set
  - with mini-batch, you'd take 5000 updates to params
  - with full batch, you'd update params 1 time with each epoch
