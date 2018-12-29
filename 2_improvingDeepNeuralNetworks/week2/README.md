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






