import csv
import numpy as np
import pandas as pd

def import_data():
    train_X = np.genfromtxt("train_X_lg_v2.csv", delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt("train_Y_lg_v2.csv", delimiter=',', dtype=np.float64)
    return train_X,train_Y

#Get Train data for class
def get_train_data_for_class(train_X, train_Y, class_label):
    class_X = np.copy(train_X)
    class_Y = np.copy(train_Y)
    class_y = np.where((class_Y == class_label), 1, 0)
    return class_X, class_Y

# Initialize, Let's initialize the parameters
def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    print(w.shape, b)
    return w, b

#Sigmoid
def sigmoid(Z):
    s = 1/(1 + np.exp(-Z))
    return s

def compute_cost(X, Y, W, b):
    m = len(Y)
    Z = np.dot(X,W) + b
    A = sigmoid(Z)
    A[A == 1] = 0.99999
    A[A == 0] = 0.00001
    cost = (-1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    return cost

#Updating (Learning Paramters)
def compute_gradient_of_cost_function(X, Y, W, b):
    m = len(Y)
    Z = np.dot(X, W)+b
    A = sigmoid(Z)
    dw = (1/m)*np.dot((A - Y).T,X)
    db = (1/m)*np.sum(A - Y)
    dw = dw.T
    return dw, db

def find_optimum_weight(X, copy_Y, alpha):
    previous_cost = 0
    W = np.zeros((len(X.T), 1),dtype=float)
    b = np.zeros((1,1),dtype=float)
    print(alpha)
    iter_no = 0
    while True:
        iter_no += 1
        dW, db = compute_gradient_of_cost_function(X, copy_Y, W, b)
        W = W - (alpha * dW)
        b = b - (alpha * db)
        cost = compute_cost(X, copy_Y, W, b)
        if iter_no % 100 == 0:
            print(iter_no, cost)
        if abs(cost - previous_cost) < 0.00000000001:
            break
        previous_cost = cost
    return W, b

def train_model(X,Y):
    Y = Y.reshape(len(X),1)
    classes = np.unique(Y)
    bb = []
    alpha = [0.00001, 0.009, 0.00001, 0.009]
    for class_label in classes:
        class_X, class_Y = get_train_data_for_class(X, Y, int(class_label))
        w, b = find_optimum_weight(class_X, class_Y, alpha[int(class_label)])
        if class_label != classes[0]:
            l = np.append(l, w, axis=1)
            bb.append(b)
        else:
            l = w
            bb.append(b)
    bb = np.array(bb, dtype=float)
    bb = np.reshape(bb,(1,4))
    l = np.vstack((l, bb))
    return l

def save_model(weights,weights_file_name):
    with open(weights_file_name,'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__ == "__main__":
    X,Y = import_data()
    weights = train_model(X,Y)
    print(weights.shape)
    save_model(weights, "WEIGHTS_FILE.csv")
