import pickle as pkl
from scipy.linalg import cho_factor, cho_solve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from utils import *
import sys

from kernels import RBF, LinearKernel, LaplacianKernel, PolynomialKernel
from classifiers import KernelSVC, MulticlassKernelSVC, OneVsOneKernelSVC, MultivariateKernelRidgeClassifier

preprocessing_type = 'hog+sift'
kernel_type = 'rbf'
kernel_param = 1.01
clf_type = 'ridge'
clf_arg = 0.0000001

def get_hog(X_train, X_val, X_test):
    hog_extractor = hog_feature_extractor()
    X_train_features = hog_extractor.extract_features(X_train)
    X_val_features = hog_extractor.extract_features(X_val)
    X_test_features = hog_extractor.extract_features(X_test)
    return X_train_features, X_val_features, X_test_features

def get_sift(X_train, X_val, X_test):
    sift = sift_extractor(130)
    sift.fit(X_train)
    X_train_features = sift.predict(X_train)
    X_val_features = sift.predict(X_val)
    X_test_features = sift.predict(X_test)
    return X_train_features, X_val_features, X_test_features

if __name__ == "__main__":
    print("Loading data...")
    Xtr, Ytr, Xte = read_data("data/")
    
    args = sys.argv
    if len(args) > 1:
        preprocessing_type = args[1]
        kernel_type = args[2]
        if kernel_type == 'rbf' or kernel_type == 'laplacian' or kernel_type == 'pol':
            kernel_param = float(args[3])
            clf_type = args[4]
            clf_arg = float(args[5])
        else:
            kernel_param = None
            clf_type = args[3]
            clf_arg = float(args[4])
    
    # Split the data
    X_train, X_val, Y_train, Y_val = train_test_split(Xtr, Ytr, test_size=0.2, random_state=42)

    # Data augmentation
    print("Augmenting data...")
    X_train, Y_train = flip_augmentation(X_train, Y_train, aug_ratio=1.0)
    X_train, Y_train = rotate_dataset(X_train, Y_train, ratio=1.0)

    print("Gettting gray scale images...")
    X_train_gray = gray_scale(X_train)
    X_val_gray = gray_scale(X_val)
    X_test_gray = gray_scale(Xte)

    print("Extracting features...")
    if preprocessing_type == 'hog':
        print("Extracting HOG features...")
        X_train_features, X_val_features, X_test_features = get_hog(X_train, X_val, Xte)
    elif preprocessing_type == 'sift':
        print("Extracting SIFT features...")
        X_train_features, X_val_features, X_test_features = get_sift(X_train_gray, X_val_gray, X_test_gray)
    elif preprocessing_type == 'hog+sift':
        print("Extracting HOG features...")
        X_train_hog, X_val_hog, X_test_hog = get_hog(X_train, X_val, Xte)
        print("Extracting SIFT features...")
        X_train_sift, X_val_sift, X_test_sift = get_sift(X_train_gray, X_val_gray, X_test_gray)
        X_train_features = np.concatenate((X_train_hog, X_train_sift), axis=1)
        X_val_features = np.concatenate((X_val_hog, X_val_sift), axis=1)
        X_test_features = np.concatenate((X_test_hog, X_test_sift), axis=1)
    else:
        raise ValueError("Invalid preprocessing type")
    
    print("Training classifier...")
    kernel = None
    if kernel_type == 'rbf':
        kernel = RBF(sigma=kernel_param)
    elif kernel_type == 'laplacian':
        kernel = LaplacianKernel(sigma=kernel_param)
    elif kernel_type == 'pol':
        kernel = PolynomialKernel(p=kernel_param)
    elif kernel_type == 'linear':
        kernel = LinearKernel()
    else:
        raise ValueError("Invalid Kernel type")
    
    if clf_type == 'ridge':
        clf = MultivariateKernelRidgeClassifier(kernel=kernel, lmbda=clf_arg)
    elif clf_type == 'svm':
        clf = OneVsOneKernelSVC(kernel=kernel, C=clf_arg)
    else:
        raise ValueError("Invalid classifier type")
    clf.fit(X_train_features, Y_train)

    print("Evaluating classifier...")
    Y_val_pred = clf.predict(X_val_features)
    accuracy = np.mean(Y_val_pred == Y_val)
    print(f"Validation accuracy: {accuracy:.4f}")
    
    print("Training on all data...")
    
    print("Augmenting data...")
    Xtr, Ytr = flip_augmentation(Xtr, Ytr, aug_ratio=1.0)
    Xtr, Ytr = rotate_dataset(Xtr, Ytr, ratio=1.0)
    print("Gettting gray scale images...")
    Xtr_gray = gray_scale(Xtr)

    print("Extracting features...")
    if preprocessing_type == 'hog':
        print("Extracting HOG features...")
        Xtr_features, _, _ = get_hog(Xtr, Xte, Xte)
    elif preprocessing_type == 'sift':
        print("Extracting SIFT features...")
        Xtr_features, _, _ = get_sift(Xtr_gray, X_test_gray, X_test_gray)
    elif preprocessing_type == 'hog+sift':
        print("Extracting HOG features...")
        Xtr_hog, _, _ = get_hog(Xtr, Xte, Xte)
        print("Extracting SIFT features...")
        Xtr_sift, _, _ = get_sift(Xtr_gray, X_test_gray, X_test_gray)
        Xtr_features = np.concatenate((Xtr_hog, Xtr_sift), axis=1)
    else:
        raise ValueError("Invalid preprocessing type")
    
    print("Training classifier...")
    if clf_type == 'ridge':
        clf = MultivariateKernelRidgeClassifier(kernel=kernel, lmbda=clf_arg)
    elif clf_type == 'svm':
        clf = OneVsOneKernelSVC(kernel=kernel, C=clf_arg)
    else:
        raise ValueError("Invalid classifier type")
    
    clf.fit(Xtr_features, Ytr)

    print("Making predictions...")
    Yte = clf.predict(X_test_features)
    Yte = {'Prediction' : Yte} 
    dataframe = pd.DataFrame(Yte) 
    dataframe.index += 1 
    dataframe.to_csv('./Yte_pred_start.csv',index_label='Id')