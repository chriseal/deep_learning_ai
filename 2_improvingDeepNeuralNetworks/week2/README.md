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

- 






