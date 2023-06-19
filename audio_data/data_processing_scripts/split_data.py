import numpy as np

def split(data_path, label_path, train_split_percent):
    data = np.load(data_path)
    labels = np.load(label_path)
    length = data.shape[0]
    split_point = int(length*train_split_percent)
    training_data = data[:split_point]
    training_labels = labels[:split_point]
    test_data = data[split_point:]
    test_labels = labels[split_point:]

    return training_data,training_labels,test_data,test_labels