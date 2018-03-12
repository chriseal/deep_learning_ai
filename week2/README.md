# Week 2

## Logistic Regression as a Neural Network

- Prediction formula:
- Loss function:
- Cost function:
- Gradient descent
![img](https://github.com/chriseal/deep_learning_ai/blob/master/week2/one%20step%20of%20gradient%20descent%20pseudo%20code.png)


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

- 
