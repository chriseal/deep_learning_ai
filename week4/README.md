# Week 4 - Deep network

## Notation in a deep layer

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week4/deep_nn_notation.png)

### Forward propagation in a deep network

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week4/forward_prop_deep.png)

## Debugging: Dimensions

- one training example
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week4/dimensions_one_training_example.png)
- vectorized example
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week4/vectorized_implementation.png)

## Why deep representations?

- deep networks work better than big shallow representations
- go from simple predictions to complex predictions as you move deeper in the hidden layers
- circuit theory and deep learning - there are functions you can compute with a 'small' L-layer deep NN that shallower network require exponentially more hidden units to compute
- When starting on a problem, Andrew Ng starts with logistic regression, then 1 hidden layer, 2 hidden layers, and generally treats the number of hidden layers as another parameter to tune (rather than just assuming a very deep network)

## Building blocks - forward and backward

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week4/forward_and_back_io.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week4/forward_and_back_io_2.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week4/generalized_backward_prop.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week4/backward_ex.png)
