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

## Derivations of activation functions

### Derivative of Sigmoid function:

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/week3/derivative_of_sigmoid.png)
- shorthand for derivative of g(z) ==> g'(z)
- slope of sigmoid is: sigmoid(z)\*(1-sigmoid(z))
  
### Derivative of tanh activation function

- = 1 - tanh(z)^2

### ReLU and Leaky ReLU

- ReLU: g'(z) = 0 if z < 0; 1 if z >= 0;
- Leaky ReLU: g'(z) = 0.01 if z < 0; 1 if z >= 0;

### Gradient Descent for Neural Networks

- dimensions:
  - given Parameters: 
    - w<sup>[1]</sup>, b<sup>[1]</sup>; w<sup>[2]</sup>, b<sup>[2]</sup>
  - and given layers: n<sup>[0]</sup> (input features), n<sup>[1]</sup> (hidden layer), n<sup>[2]</sup> (output layer, e.g. = 1)
  - Dimensions of parameters:
    - w<sup>[1]</sup>: (n[1], n[0])
    - b<sup>[1]</sup>: (n[1], 1)
    - w<sup>[2]</sup>: (n[2], n[1])
    - b<sup>[2]</sup>: (n[2], 1)
- initialize parameters randomly
![img](https://github.com/chriseal/deep_learning_ai/blob/master/week3/gradient_descent_dimensions.png)
- Formulaes for computing derivatives:
![img](https://github.com/chriseal/deep_learning_ai/blob/master/week3/formulas_for_computing_derivatives.png)

  
  
