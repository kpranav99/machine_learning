import sys
import csv
import numpy as np
from validate import validate
from train import sigmoid

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights

def predict_target_values(test_X, weights):
    b = weights[0]
    w = weights[1:]
    b = np.reshape(b,(1,4))
    pred_Y =[]
    for x in test_X:
        max_h = 0
        label = -1
        for i in range(4):
            h = sigmoid(np.dot(x, w[:,i] ) + b[0][i])
            
            if h > max_h:
                max_h = h
                label = i
        pred_Y.append(label)
        
    pred_Y = np.array(pred_Y)
    pred_Y = np.reshape(pred_Y,(len(test_X),1))
    return pred_Y

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_lg.csv")

if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    validate(test_X_file_path, actual_test_Y_file_path="test_Y_lg_v2.csv")
