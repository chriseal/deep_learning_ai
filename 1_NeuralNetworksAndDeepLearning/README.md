# Week 2

## Logistic Regression as a Neural Network

### Logistic Regression Gradient Descent - one row

- "da" variable is dL(a,y)/da (derivative of Loss function with respect to predicted value); a = y<sup>^</sup>
  - -(y/a) + (1-y)/(1-a)
- "dz" variable is dL(a,y)/dz
  - dz = a - y = dL/da\*da/dz
- dL/dw_1 = "dw_1" = x_1 \* dz, dL/dw_2 = "dw_2" = x_2 \* dz
  - w_1 := w_1 - alpha \* dw_1; w_2 := w_2 - alpha \* dw_2;
- db = dz
  - b := b - alpha \* db

### Logistic Regression Gradient Descent - m training examples

- Prediction formula:
  - y<sup>^</sup> = sigmoid(w<sup>T</sup>\*x + b)
  - sigmoid = 1 / (1 + e<sup>-z</sup>)
- Loss (error) function: applied to just a single training example
  - squared error is non-convex, so...
  - L(y<sup>^</sup>, y) = - ( y \* log y<sup>^</sup> + (1 - y) \* log(1 - y<sup>^</sup>) )
- Cost function: cost of parameters (all rows)
  - J(w, b) = 1/m * sum(L over all rows) = -1/m * sum_over_m( y_i \* log y_i<sup>^</sup> + (1 - y_i) \* log(1 - y_i<sup>^</sup>)
- Gradient descent
![img](https://github.com/chriseal/deep_learning_ai/1_NeuralNetworksAndDeepLearning/blob/master/week2/one%20step%20of%20gradient%20descent%20pseudo%20code.png)

## Vectorization

- the art of replacing for loops with vectors
- np.dot(w,x) is w<sup>t</sup>\*x, but much faster
- both GPU and CPU have parallelization instructions
  - SIMD instructions: single instruction multiple data
- vectorization allows for much easier parallelization
- whenever possible, avoid explicit for loops!

### more vectorization examples

- u = A\*v ==> u = np.dot(A, v)
- apply exponential operation on v ==> u = np.exp(v)
- elementwise operations in numpy (examples)
  - np.log(v), np.abs(v), np.maximum(v, 0)
  - v ** 2, 1/v

### Vectorizing Logistic Regression

- remember to define X's shape as (columns, rows) (i.e. transposed from standard sheet)
- w(sup)t(/sup)\*X+[b b ... b]
  - b is 1 x m size
  - 1 x n_x \* n_x x m
  - = [ w<sup>t</sup>\*x<sup>(1)</sup> + w<sup>t</sup>\*x<sup>(2)</sup> + ... + w<sup>t</sup>\*x<sup>(n)</sup> ] + [ b b ... b]
  - Z = np.dot(x.T, x) + b  # b (1,1)
  - Z is 1 x m matrix that contains the loss for each row
  - Cost / A = sigmoid(Z)

### Vectorizing Logistic Regression's Gradient Output

- A (1 x m); Y (1 x m)
- dZ = A - Y = [a1- y1 , a2 - y2, ... ]
- db = 1/m \* np.sum(dZ)
- dw = 1/m \* X \* dZ<sup>T</sup>
- Gradient descent in logistic regression:
  - ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week2/vectorized_gradient_descent_logistic_regression.png)

### Broadcasting

- .reshape is constant time (very fast)
- (m, n) + (1, n)
  - python copies (1,n) m times -> size ends up as (m, n)
- (m, n) + (m, 1)
  - python copies (m, 1) n times -> size ends up as (m, n)
- (m, n) +-\*/ (1, n) <= python copies (1, n) m times (into (m, n)) and applies it elementwise
- (m, n) +-\*/ (m, 1) <= python copies (1, n) m times (into (m, n)) and applies it elementwise
- (m, 1) \* Real_num <= copies Real_num m times

### tips and tricks to reduce errors with python/numpy vectors

- avoid vector shape (m, ); Rank 1 array
  - instead a = np.random.randn(5, 1)
  - column vector: (m, 1)
  - row vector: (1, n)
- add in assertions for shape
- a = a.reshape((5,1))

### Jupyter/iPython notebooks

- Run cell: Shift + Enter <- execute code block
- Kernel -> restart ; if any issues

### Explanation of Logistic Regression Cost Function

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week2/logistic_cost_function_explanation.png)
- add in log and keep interpretation the same
- by minimizing cost, we're really carrying out maximum likelihood estimation


# Week 3

- a<sup>[0]</sup> represents feature inputs
- a<sup>[1]</sup> represents the array of outputs from hidden layer 1
  - a<sup>[0]</sup><sub>1</sub> represents output of first node
- w<sup>[1]</sup>: weights for first hidden layer, shape is (num nodes x num inputs)
- b<sup>[1]</sup>: constants for first hidden layer, shape is (num nodes x 1)

## Computing a neural network's output - Single training example

- ![img](https://github.com/chriseal/deep_learning_ai/1_NeuralNetworksAndDeepLearning/blob/master/week3/neural_network_matrices.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week3/matrix_definitions.png)
- NN in four lines of code

## Computing a neural network's output - Multiple training examples

- a<sup>[2]</sup><sup>(i)</sup> - layer 2, training example i
- for X, Z, and A...
  - horizontally: training example index
  - vertically: node index (from top to bottom)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week3/multiple_training_examples.png)

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

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week3/derivative_of_sigmoid.png)
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

### Backpropogation intuition
![img](https://github.com/chriseal/deep_learning_ai/blob/master/week3/logistic_regression_backprop_formula.png)
![img](https://github.com/chriseal/deep_learning_ai/blob/master/week3/backprop2lyrRaw.png)
![img](https://github.com/chriseal/deep_learning_ai/blob/master/week3/backprop2lyrClean.png)

### Randomized Initialization

- initializing bias terms to 0 actually is okay
- but initializing W's to 0 could be a problem
- hidden units are computing exactly the same function (Symmetric), so when you perform the weight update, all updates will be the same
- W<sup>1</sup> initialized as np.random.randn((2,2)) * 0.01
  - need to initialize to small random numbers, especially if we're using tanh or sigmoid activation functions
  - if doing binary classification and output is a logistic function, this could still effect your result even if using ReLU
  - if very deep network, might want to choose a number other than 0.01
- b<sup>1</sup> initialized as np.zero((2,1))

## Heros of Deep Learning with Ian Goodfellow

- created GANs, came home from a bar and coded first one at midnight after realizing how to do it by arguing with a friend


# Week 4 - Deep network

## Notation in a deep layer

- ![img](https://github.com/chriseal/deep_learning_ai/1_NeuralNetworksAndDeepLearning/blob/master/week4/deep_nn_notation.png)

### Forward propagation in a deep network

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week4/forward_prop_deep.png)

## Debugging: Dimensions

- one training example
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week4/dimensions_one_training_example.png)
- vectorized example
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week4/vectorized_implementation.png)

## Why deep representations?

- deep networks work better than big shallow representations
- go from simple predictions to complex predictions as you move deeper in the hidden layers
- circuit theory and deep learning - there are functions you can compute with a 'small' L-layer deep NN that shallower network require exponentially more hidden units to compute
- When starting on a problem, Andrew Ng starts with logistic regression, then 1 hidden layer, 2 hidden layers, and generally treats the number of hidden layers as another parameter to tune (rather than just assuming a very deep network)

## Building blocks - forward and backward

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week4/forward_and_back_io.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week4/forward_and_back_io_2.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week4/generalized_backward_prop.png)
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week4/backward_ex.png)

## Parameters vs. Hyperparameters

- Parameters: W<sup>[1]</sup>, b<sup>[1]</sup>, W<sup>[2]</sup>, b<sup>[2]</sup>
- Hyperparameters:
  - parameters that determine the real parameters
  - learning rate - alpha
    - determines how parameters evolve
  - # iterations
  - # hidden layers / L
  - # hidden units, n<sup>[1]</sup>, n<sup>[2]</sup>
  - choice of activation function
  - Later: momentum, minibatch size, regularizations, ...
- so many hyperparameters in deep learning
- applied deep learning is a very empirical process
- cost J vs. # iterations - to plot at different alpha levels to determine best alpha
- try out range of values and see what works

## What does deep learning have to do with the brain?

- not a whole lot.. ha


