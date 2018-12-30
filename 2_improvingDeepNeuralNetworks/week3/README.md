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













