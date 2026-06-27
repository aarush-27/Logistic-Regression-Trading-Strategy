import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def linear(X,w,b):
    return np.dot(X,w)+b

def initialise_parameters(n_features):
    w = np.zeros((n_features,1))
    b = 0
    return w,b

def predict_probability(X,w,b):
    z = linear(X,w,b)
    p = sigmoid(z)
    return p

X = np.array([[0.02],
              [-0.01],
              [0.05]])

def compute_loss(y,p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    m = len(y)
    loss = -(1/m)*np.sum(
        y*np.log(p)+
        (1-y)*np.log(1-p))
    return loss

def compute_gradients(X,y,p):
    m = len(y)
    error = p-y
    dw = (1/m)*np.dot(X.T, error)
    db = (1/m)*np.sum(error)
    return dw,db

def update_parameters(w, b, dw, db, learning_rate):

    w = w - learning_rate * dw

    b = b - learning_rate * db

    return w, b

losses = []

def train(X, y, learning_rate = 0.01, epochs = 1000):
    n_features = X.shape[1]
    w,b = initialise_parameters(n_features)
    for epoch in range(epochs):
        p = predict_probability(X,w,b)
        loss = compute_loss(y,p)
        dw,db = compute_gradients(X, y, p)
        w,b = update_parameters(w,b,dw,db,learning_rate)
        losses.append(loss)
    return w,b,losses

def predict(X,w,b):
    p = predict_probability(X,w,b)
    return (p>0.5).astype(int)