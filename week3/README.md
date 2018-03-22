# Week 3 

- a<sup>[0]</sup> represents feature inputs
- a<sup>[1]</sup> represents the array of outputs from hidden layer 1
  - a<sup>[0]</sup><sub>1</sub> represents output of first node
- w<sup>[1]</sup>: weights for first hidden layer, shape is (num nodes x num inputs)
- b<sup>[1]</sup>: constants for first hidden layer, shape is (num nodes x 1)

## Computing a neural network's output - Single training example

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week3/neural_network_matrices.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week3/matrix_definitions.png)
- NN in four lines of code

## Computing a neural network's output - Multiple training examples

- a<sup>[2]</sup><sup>(i)</sup> - layer 2, training example i
- for X, Z, and A...
  - horizontally: training example index
  - vertically: node index (from top to bottom)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week3/multiple_training_examples.png)

## Activation functions

- sigmoid is what we've seen so far
- tanh almost always works better than sigmoid
  - shifted version of sigmoid where mean of data is closer to 0
  - makes learning in the second layer easier
  - tanh = ( e<sup>z</sup>-e<sup>-z</sup> ) / ( e<sup>z</sup>+e<sup>-z</sup> )
  - output layer is still sigmoid
  - if z is very large or small, slope approaches 0
  - still used sometimes
- Rectified Linear Unit (ReLU)
  - increasingly the default choice for activation unit
  - a = max(0, z)
  - derivative is 1 if a > 0, 0 if < 0, not well defined at a == 0
- Leaky ReLU
  - a = max(0.01\*z, z) <- could try different slopes at < 0 (0.01 here)
- test different choices and see how they perform (not one obvious choice across all problems)

## Why you need non-linear activation functions

- if you use linear activation functions, then the model is just outputting a linear function of the input
- if you're conducting regression, then the output layer will probably have a linear activation function (e.g. predicting housing prices... actually, a ReLU would actually make some sense, because housing prices cannot be negative)


  
