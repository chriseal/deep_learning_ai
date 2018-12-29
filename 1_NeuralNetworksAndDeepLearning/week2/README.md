# Week 2

## Logistic Regression as a Neural Network

### Logistic Regression Gradient Descent - one row

- "da" variable is dL(a,y)/da (derivative of Loss function with respect to predicted value); a = y<sup>^</sup>
  - -(y/a) + (1-y)/(1-a)
- "dz" variable is dL(a,y)/dz
  - dz = a - y = dL/da\*da/dz
- dL/dw_1 = "dw_1" = x_1 \* dz, dL/dw_2 = "dw_2" = x_2 \* dz
  - w_1 := w_1 - alpha \* dw_1; w_2 := w_2 - alpha \* dw_2;
- db = dz
  - b := b - alpha \* db

### Logistic Regression Gradient Descent - m training examples

- Prediction formula:
  - y<sup>^</sup> = sigmoid(w<sup>T</sup>\*x + b)
  - sigmoid = 1 / (1 + e<sup>-z</sup>)
- Loss (error) function: applied to just a single training example
  - squared error is non-convex, so...
  - L(y<sup>^</sup>, y) = - ( y \* log y<sup>^</sup> + (1 - y) \* log(1 - y<sup>^</sup>) )
- Cost function: cost of parameters (all rows)
  - J(w, b) = 1/m * sum(L over all rows) = -1/m * sum_over_m( y_i \* log y_i<sup>^</sup> + (1 - y_i) \* log(1 - y_i<sup>^</sup>)
- Gradient descent
![img](https://github.com/chriseal/deep_learning_ai/1_NeuralNetworksAndDeepLearning/blob/master/week2/one%20step%20of%20gradient%20descent%20pseudo%20code.png)

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
- Gradient descent in logistic regression:
  - ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week2/vectorized_gradient_descent_logistic_regression.png)

### Broadcasting

- .reshape is constant time (very fast)
- (m, n) + (1, n)
  - python copies (1,n) m times -> size ends up as (m, n)
- (m, n) + (m, 1)
  - python copies (m, 1) n times -> size ends up as (m, n)
- (m, n) +-\*/ (1, n) <= python copies (1, n) m times (into (m, n)) and applies it elementwise
- (m, n) +-\*/ (m, 1) <= python copies (1, n) m times (into (m, n)) and applies it elementwise
- (m, 1) \* Real_num <= copies Real_num m times

### tips and tricks to reduce errors with python/numpy vectors

- avoid vector shape (m, ); Rank 1 array
  - instead a = np.random.randn(5, 1)
  - column vector: (m, 1)
  - row vector: (1, n)
- add in assertions for shape
- a = a.reshape((5,1))

### Jupyter/iPython notebooks

- Run cell: Shift + Enter <- execute code block
- Kernel -> restart ; if any issues

### Explanation of Logistic Regression Cost Function

- ![img](https://github.com/chriseal/deep_learning_ai/blob/master/1_NeuralNetworksAndDeepLearning/week2/logistic_cost_function_explanation.png)
- add in log and keep interpretation the same
- by minimizing cost, we're really carrying out maximum likelihood estimation
