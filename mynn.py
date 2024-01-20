import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2
K = 3
X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype="uint8")
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0,1,N)
    t = np.linspace(j*4, (j+1)*4, N)
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
fig = plt.figure(figsize=(16, 16))
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

y_one_hot = np.eye(K)[y]
y = y_one_hot.T
X = X.T

def init_params(X, hidden1, hidden2, y):
    n_input = X.shape[0]
    n_output = y.shape[0]
    W1 = np.random.randn(hidden1, n_input) * np.sqrt(1 / n_input)
    W2 = np.random.randn(hidden2, hidden1) * np.sqrt(1 / hidden1)
    W3 = np.random.randn(n_output, hidden2) * np.sqrt(1 / hidden2)

    b1 = np.zeros((hidden1, 1))
    b2 = np.zeros((hidden2, 1))
    b3 = np.zeros((n_output, 1))
    return W1, W2, W3, b1, b2, b3

def forward(X, W, b):
    Z = np.dot(W, X) + b
    return Z

def sigmoid(Z):
    #Z = Z - np.max(Z)  # Logit shift
    A = 1 / (1 + np.exp(-Z))
    return A

def sigmoid_backward(A):
    dA = A * (1 - A)
    return dA

def relu(Z):
    A = np.maximum(Z, 0)
    return A

def relu_backward(dA, A):
    dA[A <= 0] = 0 
    return dA

def linear(Z):
    return Z

def linear_backward(A):
    return A

def softmax(Z):
    exp_Z = np.exp(Z- np.max(Z, axis=1, keepdims=True))
    y_hat = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    return y_hat

def feedforward_propagation(X, activation1, activation2, activation3, W1, b1, W2, b2, W3, b3):
    Z1 = forward(X, W1, b1)
    if activation1 == "sigmoid":
        A1 = sigmoid(Z1)
    elif activation1 == "relu":
        A1 = relu(Z1)
    elif activation1 == "linear":
        A1 = linear(Z1)
    
    Z2 = forward(A1, W2, b2)
    if activation2 == "sigmoid":
        A2 = sigmoid(Z2)
    elif activation2 == "relu":
        A2 = relu(Z2)
    elif activation2 == "linear":
        A2 = linear(Z2)
    
    Z3 = forward(A2, W3, b3)
    
    if activation3 == "sigmoid":
        y_hat = sigmoid(Z3.T)
    elif activation3 == "relu":
        y_hat = relu(Z3.T)
    elif activation3 == "linear":
        y_hat = linear(Z3.T)
    elif activation3 == "softmax":
        y_hat = softmax(Z3.T)
    
    return y_hat, A1, A2

def binary_ce(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
    
    logprobs = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    loss = np.sum(logprobs) / y.shape[0]
    return loss

def binary_ce_backward(y_hat, y):
    m = y.shape[0]
    J = y.shape[1]
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)

    dZ3 = (-(y / y_hat - (1 - y) / (1 - y_hat)) / J) / m
    return dZ3

def mse(y_hat, y):
    loss = np.mean((y - y_hat)**2, axis=-1)
    loss = np.sum(((y - y_hat)**2) / y.shape[0])
    return loss

def mse_backward(y_hat, y):
    m = y.shape[0]
    J = y.shape[1]

    dZ3 = (-2 * (y - y_hat) / J) / m
    return dZ3

def mae(y_hat, y):
    loss = np.sum(np.mean(np.abs(y - y_hat), axis=-1) / y.shape[0])
    return loss

def mae_backward(y_hat, y):
    m = y.shape[0]
    J = y.shape[1]

    dZ3 = (np.sign(y - y_hat) / J) / m
    return dZ3

def categorical_ce(y_hat, y):
    loss = -np.sum(y * np.log(np.maximum(y_hat, 1e-15))) / y.shape[0]
    return loss

def softmax_ce_backward(y_hat, y):
    dZ3 = (y_hat - y) / y.shape[0]
    return dZ3

def backward_propagation(dA3, activation1, activation2, activation3, X, W2, W3, A1, A2, A3):
    m = X.shape[1]
    if activation3 == "sigmoid":
        dZ3 = sigmoid_backward(dA3)
    elif activation3 == "relu":
        dZ3 = relu_backward(dA3, A3)
    elif activation3 == "linear":
        dZ3 = linear_backward(dA3)
    elif activation3 == "softmax":
        dZ3 = dA3
    
    if activation2 == "sigmoid":
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = sigmoid_backward(dA2)
    elif activation2 == "relu":
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = relu_backward(dA2, A2)
    elif activation2 == "linear":
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = linear_backward(dA2)

    if activation1 == "sigmoid":
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = sigmoid_backward(dA1)
    elif activation1 == "relu":
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = relu_backward(dA1, A1)
    elif activation1 == "linear":
        dA1 = np.dot(W3.T, dZ3)
        dZ1 = linear_backward(dA2)

    
    dW3 =np.dot(dZ3, A2.T)
    dW2 =np.dot(dZ2, A1.T)
    dW1 =np.dot(dZ1, X.T)
    db1 =np.sum(dZ1, axis=1, keepdims=True)
    db2 =np.sum(dZ2, axis=1, keepdims=True)
    db3 =np.sum(dZ3, axis=1, keepdims=True)

    return dW1, dW2, dW3, db1, db2, db3

def momentum(vdW1, vdW2, vdW3, vdb1, vdb2, vdb3, dW1, dW2, dW3, db1, db2, db3, beta1):
    vdW1 = beta1*vdW1 + (1-beta1)*dW1
    vdW2 = beta1*vdW2 + (1-beta1)*dW2
    vdW3 = beta1*vdW3 + (1-beta1)*dW3
    vdb1 = beta1*vdb1 + (1-beta1)*db1
    vdb2 = beta1*vdb2 + (1-beta1)*db2
    vdb3 = beta1*vdb3 + (1-beta1)*db3

    return vdW1, vdW2, vdW3, vdb1, vdb2, vdb3

def adagrad(cacheW1, cacheW2, cacheW3, cacheb1, cacheb2, cacheb3, dW1, dW2, dW3, db1, db2, db3):
    epsilon = 10e-8
    cacheW1 += dW1 ** 2
    cacheW2 += dW2 ** 2
    cacheW3 += dW3 ** 2
    cacheb1 += db1 ** 2
    cacheb2 += db2 ** 2
    cacheb3 += db3 ** 2
    dW1 = dW1 / (np.sqrt(cacheW1) + epsilon)
    dW2 = dW2 / (np.sqrt(cacheW2) + epsilon)
    dW3 = dW3 / (np.sqrt(cacheW3) + epsilon)
    db1 = db1 / (np.sqrt(cacheb1) + epsilon)
    db2 = db2 / (np.sqrt(cacheb2) + epsilon)
    db3 = db3 / (np.sqrt(cacheb3) + epsilon)

    return  cacheW1, cacheW2, cacheW3, cacheb1, cacheb2, cacheb3, dW1, dW2, dW3, db1, db2, db3

def rmsprop(sdW1, sdW2, sdW3, sdb1, sdb2, sdb3, dW1, dW2, dW3, db1, db2, db3, beta2):
    epsilon = 1e-8
    sdW1 = beta2*sdW1 + (1-beta2) * np.square(dW1)
    sdW2 = beta2*sdW2 + (1-beta2) * np.square(dW2)
    sdW3 = beta2*sdW3 + (1-beta2) * np.square(dW3)
    sdb1 = beta2*sdb1 + (1-beta2) * np.square(db1)
    sdb2 = beta2*sdb2 + (1-beta2) * np.square(db2)
    sdb3 = beta2*sdb3 + (1-beta2) * np.square(db3)

    dW1 = dW1 / (np.sqrt(sdW1) + epsilon)
    dW2 = dW2 / (np.sqrt(sdW2) + epsilon)
    dW3 = dW3 / (np.sqrt(sdW3) + epsilon)
    db1 = db1 / (np.sqrt(sdb1) + epsilon)
    db2 = db2 / (np.sqrt(sdb2) + epsilon)
    db3 = db3 / (np.sqrt(sdb3) + epsilon)

    return  sdW1, sdW2, sdW3, sdb1, sdb2, sdb3, dW1, dW2, dW3, db1, db2, db3

def adam(sdW1, sdW2, sdW3, sdb1, sdb2, sdb3, vdW1, vdW2, vdW3, vdb1, vdb2, vdb3, dW1, dW2, dW3, db1, db2, db3, beta1, beta2, t):
    
    vdW1, vdW2, vdW3, vdb1, vdb2, vdb3 = momentum(vdW1, vdW2, vdW3, vdb1, vdb2, vdb3, dW1, dW2, dW3, db1, db2, db3, beta1)
    sdW1, sdW2, sdW3, sdb1, sdb2, sdb3, _, _, _, _, _, _ = \
    rmsprop(sdW1, sdW2, sdW3, sdb1, sdb2, sdb3, dW1, dW2, dW3, db1, db2, db3, beta2)
    epsilon = 10e-8

    vdW1c = vdW1 / (1-beta1**t)
    vdW2c = vdW2 / (1-beta1**t)
    vdW3c = vdW3 / (1-beta1**t)
    vdb1c = vdb1 / (1-beta1**t)
    vdb2c = vdb2 / (1-beta1**t)
    vdb3c = vdb3 / (1-beta1**t)
    
    sdW1c = sdW1 / (1-beta2**t)
    sdW2c = sdW2 / (1-beta2**t)
    sdW3c = sdW3 / (1-beta2**t)
    sdb1c = sdb1 / (1-beta2**t)
    sdb2c = sdb2 / (1-beta2**t)
    sdb3c = sdb3 / (1-beta2**t)

    dW1 = vdW1c / (np.sqrt(sdW1c) + epsilon)
    dW2 = vdW2c / (np.sqrt(sdW2c) + epsilon)
    dW3 = vdW3c / (np.sqrt(sdW3c) + epsilon)
    db1 = vdb1c / (np.sqrt(sdb1c) + epsilon)
    db2 = vdb2c / (np.sqrt(sdb2c) + epsilon)
    db3 = vdb3c / (np.sqrt(sdb3c) + epsilon)
    
    return dW1, dW2, dW3, db1, db2, db3, vdW1, vdW2, vdW3, vdb1, vdb2, vdb3, sdW1, sdW2, sdW3, sdb1, sdb2, sdb3

def update(W1, W2, W3, b1, b2, b3, dW1, dW2, dW3, db1, db2, db3, alpha, vdW1, vdW2, vdW3, vdb1, vdb2, vdb3,
           cacheW1, cacheW2, cacheW3, cacheb1, cacheb2, cacheb3, sdW1, sdW2, sdW3, sdb1, sdb2, sdb3,
           optimizer, beta1 , beta2, t):
    
    if optimizer == "momentum":
        dW1, dW2, dW3, db1, db2, db3 = momentum(vdW1, vdW2, vdW3, vdb1, vdb2, vdb3, dW1, dW2, dW3, db1, db2, db3, beta1)
       
    elif optimizer == "adagrad":
        cacheW1, cacheW2, cacheW3, cacheb1, cacheb2, cacheb3, dW1, dW2, dW3, db1, db2, db3 = adagrad(cacheW1, cacheW2, cacheW3, cacheb1, cacheb2, cacheb3, dW1, dW2, dW3, db1, db2, db3)

    elif optimizer == "rmsprop":
        sdW1, sdW2, sdW3, sdb1, sdb2, sdb3, dW1, dW2, dW3, db1, db2, db3 = \
        rmsprop(sdW1, sdW2, sdW3, sdb1, sdb2, sdb3, dW1, dW2, dW3, db1, db2, db3, beta2)

    elif optimizer == "adam":
        dW1, dW2, dW3, db1, db2, db3, vdW1, vdW2, vdW3, vdb1, vdb2, vdb3, sdW1, sdW2, sdW3, sdb1, sdb2, sdb3 = \
        adam(sdW1, sdW2, sdW3, sdb1, sdb2, sdb3, vdW1, vdW2, vdW3, vdb1, vdb2, vdb3, dW1, dW2, dW3, db1, db2, db3, beta1, beta2, t+1)

    W1 -= alpha * dW1
    W2 -= alpha * dW2
    W3 -= alpha * dW3
    b1 -= alpha * db1
    b2 -= alpha * db2
    b3 -= alpha * db3
    return W1, W2, W3, b1, b2, b3, vdW1, vdW2, vdW3, vdb1, vdb2, vdb3, cacheW1, cacheW2,\
           cacheW3, cacheb1, cacheb2, cacheb3, sdW1, sdW2, sdW3, sdb1, sdb2, sdb3

def plot_loss(loss_history):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(loss_history)
    plt.title('Kayıp (Loss) Değişimi')
    plt.xlabel('İterasyon')
    plt.ylabel('Kayıp')
    plt.show()

def accuracy(y_hat, y):
    max_indices = np.argmax(y_hat, axis=1)

    y_one_hot = np.zeros_like(y_hat)
    y_one_hot[np.arange(len(y_hat)), max_indices] = 1

    accuracy = (np.sum(y_hat * y) / y.shape[0]) * 100
    return accuracy

def initializer(X, y, hidden1, hidden2, alpha, activation1, activation2, activation3, iteration, loss_function,
                optimizer, batch_size, beta1, beta2):
    
    indices = np.arange(X.shape[1])
    np.random.shuffle(indices)
    X = X[:, indices]
    y = y[:, indices]
    
    W1, W2, W3, b1, b2, b3 = init_params(X, hidden1, hidden2, y)
    vdW1, vdW2, vdW3, vdb1, vdb2, vdb3 = 0, 0, 0, 0, 0, 0
    cacheW1, cacheW2, cacheW3, cacheb1, cacheb2, cacheb3 = 0, 0, 0, 0, 0, 0
    sdW1, sdW2, sdW3, sdb1, sdb2, sdb3 = 0, 0, 0, 0, 0, 0
    loss_history = []
    
    for i in range(iteration):
        for k in range(X.shape[1] // batch_size):
            X_batch = X[:, k * batch_size : (k + 1) * batch_size]    
            y_batch = y[:, k * batch_size : (k + 1) * batch_size]
            
            if X.shape[1] % batch_size != 0 and k + 1 == X.shape[1] // batch_size:
                X_batch = X[:, (k + 1) * batch_size :]    
                y_batch = y[:, (k + 1) * batch_size :]
            
            
            y_hat, A1, A2 = feedforward_propagation(X_batch, activation1, activation2, activation3, W1, b1, W2, b2, W3, b3)

            if loss_function == "categorical_ce":
                loss = categorical_ce(y_hat, y_batch.T)
                dA3 = softmax_ce_backward(y_hat, y_batch.T)

            elif loss_function == "binary_ce":
                loss = binary_ce(y_hat, y_batch.T)
                dA3 = binary_ce_backward(y_hat, y_batch.T)

            elif loss_function == "mse":
                loss = mse(y_hat, y_batch.T)
                dA3 = mse_backward(y_hat, y_batch.T)

            elif loss_function == "mae":
                loss = mae(y_hat, y_batch.T)
                dA3 = mae_backward(y_hat, y_batch.T)

            dW1, dW2, dW3, db1, db2, db3 = backward_propagation(dA3.T, activation1, activation2, activation3, X_batch, W2, W3, A1, A2, y_hat)
        
            W1, W2, W3, b1, b2, b3, vdW1, vdW2, vdW3, vdb1, vdb2, vdb3, cacheW1, cacheW2,\
            cacheW3, cacheb1, cacheb2, cacheb3, sdW1, sdW2, sdW3, sdb1, sdb2, sdb3 = update(W1, W2, W3, b1, b2, b3, dW1, dW2, dW3, db1, db2, db3, alpha, vdW1, vdW2, vdW3,
                                            vdb1, vdb2, vdb3,cacheW1, cacheW2, cacheW3, cacheb1, cacheb2, cacheb3, 
                                            sdW1, sdW2, sdW3, sdb1, sdb2, sdb3,optimizer, beta1, beta2, i)
        loss_history.append(loss)
        if (i % 1000) == 0:
            acc = accuracy(y_hat, y_batch.T)
            print(f"loss iteration {i}: {loss}   accuracy: {acc}   learning rate: {alpha}")
            
    plot_loss(loss_history)
BGD = X.shape[1]
SGD = 1

iteration = 5001
hidden1 = 32
hidden2 = 32
activation1 = "relu"
activation2 = "relu"
activation3 = "softmax" #sigmoid, relu, linear, softmax
alpha = 0.07
loss_function = "categorical_ce" #categorical_ce, binary_ce, mse, mae
optimizer = "adam" #momentum, adagrad, rmsprop, adam
batch_size = BGD #BGD, SGD
beta1 = 0.9
beta2 = 0.999

initializer(X, y, hidden1, hidden2, alpha, activation1, activation2, activation3, iteration, loss_function,
            optimizer, batch_size, beta1, beta2)
