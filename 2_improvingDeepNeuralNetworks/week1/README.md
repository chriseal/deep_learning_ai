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






