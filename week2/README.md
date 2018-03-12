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

