# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

# Week 1

## train/dev_test/test splits

- how you setup your data and parameters greatly impacts model success (even for very experienced people in deep learning)
- very iterative process
- how efficiently you iterate helps you try more ideas faster

- how to split dataset:
  - traditional: 70/30 train/dev test or 60/20/20 train/dev test/test
    - works for smaller datasets (say, <= 10,000)
  - big data:
    - with 1,000,000 examples, 10,000 test examples might be just fine...
      --> 98/1/1 train/dev test/test split
      
- Mismatched train/test datasets
  - ex: training set is cat pictures from web pages (higher res) and test set is cat pictures uploaded from your app (lower res)
  - rule of thumb: make sure dev and test come from same distribution
  
- not having a test set might be okay (only dev set) if you don't need a completely unbiased estimate of model performance

## Bias / Variance

- very important
- less of a bias / variance trade-off
  - training set error compared to dev set error
  - say training set error is much lower than dev set error
- Optimal Error / Bayes Error: ex. compared to human performance
- it's possible to get high bias in some areas and high variance in other areas in higher dimensional space

## Basic Recipe for Machine Learning

- High bias?
  - low training data performance
  - bigger network, train longer, (possibly, NN architecture)
  - try this until you get rid of bias problem
- High variance?
  - low dev set performance
  - get more data, try regularization, (possibly, NN architecture)
- "Bias / variance tradeoff"  
  - no longer as important, if you can always train a bigger network (reduce bias) or get more data (reduce variance) independently
  - much less tradeoff bw b/v in deep learning
  - if you're regularizing, a bigger network almost never hurts
  
## How to apply regularization

- tend to only regularize W and not regularize b (Ng does not regularize b, because it doesn't make much of a difference... most parameters are in W)
- L1 regularization vs L2 regularization
  - L1 regularization moves things to zero, which will "help compress" models
  - in practice, L1 doesn't help that much to compress models
  - most people use L2 regularization
- tend to iterate over lambda
- use `lambd` in python, because lambda is reserved
- use Frobenius norm of a matrix (similar to L2 norm)
  - sum of square of elements of a matrix
  - aka "Weight Decay", because it slightly reduces the values in W
- how to compute gradient with regularization?
  - dW = (from backprop) + (lambd / m) * W

## Why regularization reduces overfitting?

- setting weights close to zero effectively reduces them to zero, which is akin to dropout-ish
  - makes it a simpler network
- in practice, still uses all the hidden units but makes them have less of an impact, which makes a simper model
- if using tanh, regularization keeps the weights where the slope is higher, so training happens faster
  - tanh is nearly linear around zero.... so keeps tanh more linear --> makes model more simple
- one of the steps to debug gradient descent is to plot cost vs. # of iterations
- Ng uses L2 regularization very often

## Dropout regularization

- very powerful in addition to L2 regularization
- randomly remove nodes for each iteration, so each iteration is using a smaller forward and backprop network
- smaller networks are trained at a time
- Implementing dropout ("inverted dropout")
  - have to drop out nodes and then rescale remaining weights back up by dividing by keep probability
    --> makes expected value the same, bc you have less of a scaling problem
- making predictions at test time, don't use dropout, because weights already have dropout affected by dropout in training
  - because you've already rescaled during training with dropout
  
## Understanding Dropout / Why does dropout work?

- intuition is it can't rely on any one feature - spreads out the weights, similar effect to shrinking the weights
- dropout can be formally be shown to be adaptive
- similar effect to L2, but dropout with L2 can be even more adaptive to the scale
- can have keep_prob vary inversely with the size of each hidden layer 
  - intuition being that if hidden layer has 300 units, you can drop half of them more easily than if the hidden layer has 2 units
- technically, you can apply dropout to input layer, but in practice, this doesn't happen that often
- can only apply dropout to some layers not others to speed up training time / searches
- if you don't have a variance problem, might not be a need to use dropout
- harder to detect Cost function
  --> turn off dropout first to make sure cost function is reducing with every/almost every iteration
    - then, once you're sure about that / less likely for bugs, start implementing dropout
  
## Other regularization methods

- Data Augmentaion - adding more data / synthesizing data:
  - flip images horizontally to get more data
  - take random crops of an image
  - inexpensive way to "add more data"
  - add random permutations to digit images, for example... can use a more subtle distortion in practis
- Early Stopping:
  - plot dev set error with training set error
  --> stop iterating when dev set loss is minimized
- one downside:
  - machine learning is: 1) optimize J, 2) not overfit
    - Ng likes to focus on each step one at a time - "Orthogonalization"
  - Early stopping combines 1 and 2 in one step, which complicates the search space
  - Ng uses L2 regularization instead, is better, but takes more time bc bigger search space
  
## Normalizing inputs

- speeds up training of model
1) subtract the mean
2) normalize variance 
- use same mean and stancard deviation to scale train and test set similarly
- why normalize?
  - without normalization, cost function curve could be very spread out, which takes longer to optimize
    - might need very small learning rate (bc a lot of steps)
  - with normalization, cost function looks more symmetric
    - gradient descent can often take bigger steps

## Vanishing / Exploding gradients

- say you're training a very deep NN
- in very deep NN's, weights are kinda getting multiplied with each hidden layer 
  - extreme case: W<sup>[L]</sup> --> each individual weight will be ~weight<sub>[L-1]</sub><sup>L-1</sup>
  - derivatives can also increase/decrease exponentially
  
## Weight Initialization for Deep Networks

- can help prevent vanishing/exploding gradients
- the larger number of inputs -> the smaller each weight should be, because you're adding them up
- one thing to do: set variance of W<sub>i</sub> = 1/n
  - 2/n better for ReLu - He initialization
  - tanh - use Xavier initialization
  - just gives you a starting point, sometimes you can try this as a hyperparameter (sometimes can be helpuful)
![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week1/2wk1_scaling_weights_initialization_in_hidden_layers.png)


## Numerical approximation of gradients

- checking your derivate
- epsilon is how much you move in a direction to test/approximate gradient
- test derivative by moving both to the right and to the left
  - use a bigger triangle to test gradient to make derivative approximation more accurate
  - twice as slow as one-sided test, but Ng thinks it's worth it
- exponentially more accurate to take 2-sided difference versus 1-sided difference

## Gradient checking

- has helped Ng save lots of time
- take all W's and b's, reshape into big vector `theta`
- reshape all dW's and db's into one big vector `dtheta`
- Is `dtheta` the gradient of the slope of the cost function J? 
  - check use Euclidean distance (just square root, no squaring)
  - make sure epsilon <= 10<sup>-7</sup>, 10<sup>-5</sup> *might* be okay, 10<sup>3</sup> is *not* okay
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week1/2wk1_gradient_checking.png)


## Gradient Checking Implementation Notes

- only use in debug; turn off in model training
- find which values are very far off (could be all in db, for example)... all items very far off could be in db but dW could be just fine and vv
- remember regularization in calculating gradients
- doesn't work with dropout
- run at random initialization (when W and b are close to zero), possibly double check after some training time, when W and b have had some time to wander away from zero (Ng doesn't do this very often in practice)








# Optimization Algorithms

## Mini-batch gradient descent

- let gradient descent make some progress before iterating through entire training set
- helps speed things up with really large datasets
- say, m=5M, minibatch has 1000 m in each --> 5K minibatchs
- X<sup>{1-5000}</sup>... {} denotes minibatch index
- 1 epoch is a single pass though the training set
  - with mini-batch, you'd take 5000 updates to params
  - with full batch, you'd update params 1 time with each epoch

## Understanding mini-batch gradient descent

- if cost function goes up at any point in full-batch, your learning rate is probably too high
- mini-batch gradient descent looks more jagged (still trends downwards)
- minibatch size
  - =m : full batch; takes too long on larger training sets
  - =1 : stochastic gradient descent
    - won't ever converge, but will get close to optimum and hang around
    - probably use smaller learning rate
    - lose all speed-up from vectorization (still slow)
  - >1 and < m:
    - best of both worlds
- Rules of thumb
  - if m < ~2k: use full batch
  - typical minibatch sizes: a power of 2 is faster
    - 64, 128, 256, 512 <- most common
    - make sure mini-batch fits in CPU/GPU (otherwise, performance will drop off a cliff)
  - can be used as another hyperparameter

## Exponentially weighted averages

- needed to go faster than mini-batch
- moving average ish
- V<sub>t</sub> = Beta * theta<sub>t-1</sub> + (1-Beta) * theta<sub>t</sub>; beta is scalar
  - V<sub>t</sub> approximately averages over 1 / (1-beta)
  - if beta = 0.9, it's ~ avg over past 10 data points
  
## Understanding exponentially weighted averages

- takes very little memory, computationally cheap, 1 line of code
- not most accurate way, but much more efficient

## Bias Correction in exponentially weighted averages

- exponentially weighted averages start out too low, bc prev data points are missing
  - initial phase inaccurate
- V<sub>t</sub> / (1 - beta<sup>t</sup>)
- people don't often bother to implement bias correction, but it's an option

## Gradient descent with momentum

- compute exponentially weighted moving gradient descent
- is almost always faster
- learning rate has to be small if not using momentum, so gradient doesn't overshoot optimum
  - V<sub>dW</sub> = Beta * V<sub>dw</sub> + (1-Beta) * dW
  - V<sub>db</sub> = Beta * V<sub>db</sub> + (1-Beta) * db
  V<sub>dw</sub> is velocity, dW is acceleration, Beta is friction
- smoothes out gradient
  - W := W - alpha * V<sub>dW</sub>
  - b := b - alphba * V<sub>db</sub>
- Beta = 0.9 works very well
  - no need for bias correction since only looking at last 10 examples ish
![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week2/2wk2_momemtum_implementation_formulas.png)

## RMSprop - Root Mean Squared prop

- another optimization algorithm
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week2/2wk2_rmsprop.png)
- can be best when combined with momentum
- similar benefits to momentum

## Adam optimization algorithm

- adaptive moment estimation
- stood test of time across a variety of verticals
- B<sub>1</sub> - momentum, first moment
- B<sub>2</sub> - RMS prop, second moment
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week2/2wk2_adam_optimization.png)
- tend to use defaults for B<sub>1</sub> (0.9) and B<sub>2</sub> (0.999), Epsilon (10<sup>-8</sup>, but doesn't affect it much)
- most people use alpha as a hyperparameter but defaults on the other 

## Learning rate decay

- learning rate gets smaller as interations increase
- alpha = 1 / (1 + decay_rate * epoch_num) * alpha<sub>0</sub>
- decay_rate can be a hyperparameter, lower down the list of hyperparameters to optimize
- alpha<sub>0</sub> can be a hyperparameter, lower down the list of hyperparameters to optimize
- can use exponential decay

## The problem of local optima

- most points of 0 gradient are saddle points in higher dimensional space
- very unlikely for all 20,000 features to be at zero at the same time
- a saddle of a horse shape
- low dimensional space intuition/issues don't translate to higher dimensional spaces as far as finding local optima goes
- Plateaus are more of an issue
  - very long plateau before finding a steeper slope
  - can make learning very slow, but momentum, RMS, Adam all help with this











# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization - Week 3

## Tuning process

- number of parameters to tune is very difficult thing with NN's
- red: most important (by far), yellow: 2nd tier, purple: 3rd tier, Ng doesn't tune betas and epsilon when using Adam Optimizer
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week3/2wk3_hyperparam_priority.png)
- try random values, don't use a grid
  - hard to know which params will be most important combinations
  - if you have alpha in a gridsearch, you could be searching over another non-important parameter while keeping alpha the same (wasting time)
  - when using random sampling, you're trying more values
- Coarse to fine scheme
  - gradually reduce the search space, as hyperparameter approaches optimum
  
## Using an appropriate scale to pick hyperparameters

- alpha could be on a log scale instead of linear
  - use this code for alpha:
```
r = -4 * np.random.rand()
alpha = 10**r
```

- sampling beta for exponentially weighted averages
  - beta between 0.9...0.999 (0.9 last 10 samples, 0.999 last 1000 samples)
```
r = -2*np.random.rand() - 1 # r from -3 to -1
beta = 1-10**r
```

  - when beta close to 1, it has a huge impact on exactly what your algorithm is doing
  - this method samples more densely when beta is closer to 1, where the impact of small changes is greater

## Hyperparameters tuning in practice: Pandas vs. Caviar

- retest hyperparameters every few months
- babysitting one model:
  - watch model every day and make nudges to learning rate, etc
  - pandas have one baby at a time and take great care of it
- train many models in parallel
  - test a lot of hyperparameter settings and pick which one looks best
  - caviar: have 1000s of babies and hope some work out
- if you have enough computers, take Caviar approach
- without enough computers, take Panda approach

## Normalizing activations in a network / Batch Normalization

- very important / helpful
- normalize inputs to speed up learning (normalize X)
- normalize activations of a hidden layer to make training of following layer more efficient (normalize a<sup>L</sup>)
  - tend to normalize Z<sup>L</sup> (use this as a default choice, but some debate over which to normalize Z or A)
- don't want hidden units to always have uniform distributions (might be less predictive)
  --> add beta (no relation to optimization beta) and gamma that are tunable params
  Z~ = gamma * Z + beta
  - can use Gradient Descent, Adam, RMS Prop, etc, to update beta and gamma
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week3/2wk3_batch_norm.png)
- when using minibatch, you implement batch norm within each respective minibatch (not globally)
- when using batch norm, you don't need b constants, because of beta in batch norm
- Z<sup>L</sup>, beta<sup>L</sup>, gamma<sup>L</sup> : (n<sup>L</sup>, 1)

## Why does Batch Norm work?

- scaling features speeds up training, just like when scaling input features
- more robust to changes to weights earlier on in the network
  - this can make the model more generalizable
    - e.g. training on all black cats, but testing on colored cats
 - "covariate shift": even if ground truth function remains the same, your X may only account for part of it
  - Activations are always changing, which makes it harder for subsequent layer (covariate shift)
  - Batch Norm reduces the amount activations change which limits the covariate shift / makes it easier for batch norm
  - Activations become more stable, so deeper layers can learn more "independently" 
  - speeds up training
- Batch Norm as regularization
  - if using mini-batch, mean/variance of Z's only calculated on mini-batch
  - adds some subtractive noise (mean centering); additive noise (constant added)
  - slight regularization effect
  - with bigger mini-batch size, you reduce the noise and regularization effect
  - but intent isn't for regularization; don't use it for just regularization
  - this is similar to dropout and has a sli

## Batch Norm at test time

- at training time, batches are normalized independently with each mini-batch
- so what to do at test time?
  - estimate mean and standard deviation using exponentially weighted average across minibatches
  - could also retrain model on whole dataset, but exponentially weighted average is pretty robust, so the latter is typically done in practice
  
## Softmax Regression

- multiclass classification
- C = number of classes
- output layer has C hidden units, each of which maps to one class
- vectorized activation function
- generalization of linear regression
  - when used without any previous hidden layers: linear boundary between any two classes 
  - in NN, decision boundary can be much more complex
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week3/2wk3_softmax.png)

## Training a softmax classifier

- "hard max" is a vector of 0's and 1's
- softmax reduces to logistic regression if C == 2
- Loss function:
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week3/2wk3_softmax_cost.png)
- backpropagation:
  - dZ<sup>[L]</sup> = y_hat - y

## Deep learning frameworks

- choosing deep learning framework
  - ease of programming (research and production)
  - running speed
  - truly open (open source with good governance)
    - how much do you trust it will remain open source for a long time (vs. overseen by a company)?

## TensorFlow

- TensorFlow automatically computes backpropagation steps when computing computation graph of cost function
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week3/2wk3_tensorflow_code_example.png)






