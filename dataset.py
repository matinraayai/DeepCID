"""
Author: Matin Raayai Ardakani
Contains Tensorflow datasets used across this project.
"""
import tensorflow as tf
import numpy as np
from sklearn import preprocessing


def raman_dataset(compound=0, path='./augmented_data',
                  batch_size=300, train_fold=0.75, valid_fold=0.875):
    x = np.load(path + "/" + str(compound) + 'component.npy').astype(np.float)
    y = np.load(path + "/" + str(compound) + 'label.npy').astype(np.float)
    assert(len(x) == len(y))
    train_fold_pos = int(len(x) * train_fold)
    valid_fold_pos = int(len(x) * valid_fold)
    # Pre-processing Xs:================================================================================================
    x_train = x[:train_fold_pos]
    x_valid = x[train_fold_pos:valid_fold_pos]
    x_test = x[valid_fold_pos:]
    # standard Gaussian scalar:
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    # Pre-processing Ys:================================================================================================
    y_neg = np.ones(y.shape) - y
    y = np.concatenate((y, y_neg), axis=1)
    y_train = y[:train_fold_pos]
    y_valid = y[train_fold_pos:valid_fold_pos]
    y_test = y[valid_fold_pos:]
    # Creating tf.data.Dataset objects:=================================================================================
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_dataset, valid_dataset, test_dataset
