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
  









