# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization - Week 3

## Tuning process

- number of parameters to tune is very difficult thing with NN's
- red: most important (by far), yellow: 2nd tier, purple: 3rd tier, Ng doesn't tune betas and epsilon when using Adam Optimizer
- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/2_improvingDeepNeuralNetworks/week3/2wk3_adam_optimization.png)
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







