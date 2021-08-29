import sys
import csv
import math
import numpy as np
from validate import validate

def import_data(train_X_file_path, train_Y_file_path, test_X_file_path):
    train_X = np.genfromtxt(train_X_file_path, delimiter = ',', dtype = np.float64, skip_header = 1)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter= ',', dtype = np.float64)
    test_X = np.genfromtxt(test_X_file_path, delimiter = ',', dtype = np.float64, skip_header = 1)
    return train_X, train_Y, test_X

def compute_ln_norm_distance(vector_1, vector_2, n):
    total_sum = 0
    for i in range(len(vector_1)):
        total_sum += (abs(vector_1[i] - vector_2[i])**n)
    return (total_sum**(1/n))

def find_k_nearest_neighbors(train_X, test_example, k, n):
    D = []
    for i in range(len(train_X)):
        dist = compute_ln_norm_distance(train_X[i], test_example, n)
        D.append((dist, i))
    D.sort()
    ans = []
    for i in range(k):
        ans.append(D[i][1])
    return ans

def calculate_accuracy(predicted_Y, actual_Y):
    correct_pred = 0
    for i in range(len(actual_Y)):
        correct_pred += (predicted_Y[i] == actual_Y[i])
    return correct_pred / len(actual_Y)

def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    test_Y = []
    for test_elem_x in test_X:
      top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k, n)
      top_knn_labels = []

      for i in top_k_nn_indices:
        top_knn_labels.append(train_Y[i])
      Y_values = list(set(top_knn_labels))

      max_count = 0
      most_frequent_label = -1
      for y in Y_values:
        count = top_knn_labels.count(y)
        if(count > max_count):
          max_count = count
          most_frequent_label = y
          
      test_Y.append(int(most_frequent_label))
    return np.array(test_Y)

def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent, n):
    total_num_of_observations = len(train_X)
    train_length = math.floor((float(100 - validation_split_percent))/100 * total_num_of_observations )
    #print(train_length)
    validation_X = train_X[train_length:]
    validation_Y = train_Y[train_length:]
    train_X = train_X[0:train_length]
    train_Y = train_Y[0:train_length]

    best_k = -1
    best_accuracy = 0
    for k in range(1, train_length+1):
        predicted_Y = classify_points_using_knn(train_X, train_Y, validation_X, k, n)
        accuracy = calculate_accuracy(validation_Y, predicted_Y)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    return best_k

def write_to_csv_file(pred_Y, predicted_Y_file_path):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_path, 'w', newline = '') as file:
        wr = csv.writer(file)
        wr.writerows(pred_Y)
        file.close()

if __name__=="__main__":
    train_X_file_path = "train_X_knn.csv"
    train_Y_file_path = "train_Y_knn.csv"
    test_X_file_path = sys.argv[1]
    train_X, train_Y, test_X = import_data(train_X_file_path, train_Y_file_path, test_X_file_path)
    k = get_best_k_using_validation_set(train_X, train_Y, 30, 4)
    pred_Y = classify_points_using_knn(train_X, train_Y, test_X, k, 4)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")
    validate(test_X_file_path, actual_test_Y_file_path = "test_Y_knn.csv")
