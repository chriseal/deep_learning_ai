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
