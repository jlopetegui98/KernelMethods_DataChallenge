import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from kernels import RBF, LinearKernel
from classifiers import KernelSVC, MulticlassKernelSVC
from timeit import default_timer as timer
from datetime import timedelta
from tqdm import tqdm

def cluster_daisy_features(daisy_feature_list,number_of_clusters):
    km=MiniBatchKMeans(n_clusters=number_of_clusters,batch_size=number_of_clusters*10, verbose=1)
    km.fit(daisy_feature_list)
    return km

def prepare_features(daisy_features, hog_features,daisy_cluster_model):
    ##find to which clusters each daisy feature belongs
    daisy_features=daisy_features.reshape(-1,1)
    img_clusters=daisy_cluster_model.predict(daisy_features) 
    cluster_freq_counts=pd.DataFrame(img_clusters,columns=['cnt'])['cnt'].value_counts()
    bovw_vector=np.zeros(daisy_cluster_model.n_clusters) ##feature vector of size as the total number of clusters

    for key in cluster_freq_counts.keys():
        bovw_vector[key]=cluster_freq_counts[key]

    bovw_feature=bovw_vector/np.linalg.norm(bovw_vector)
    hog_features=hog_features/np.linalg.norm(hog_features)

    return list(bovw_feature)+list(hog_features)

if __name__ == '__main__':
    data_path = 'data/'
    Xtr,Ytr,Xte = read_data(data_path)

    #Xtr_gray = gray_scale(Xtr)
    #Xte_gray = gray_scale(Xte)

    # split the training data into a training and a validation set
    X_train, X_val, Y_train, Y_val = train_test_split(Xtr, Ytr, test_size=0.2, random_state=42, stratify=Ytr)

    # data augmentation for training
    X_train_augmented, Y_train_augmented = flip_augmentation(Xtr, Y_train)
    X_train_augmented, Y_train_augmented = rotate_dataset(X_train_augmented, Y_train_augmented)

    # extract DAISY features
    DAISY = daisy_feature_extractor()
    print("Extracting DAISY features")
    start = timer()
    X_train_daisy = DAISY.extract_features(X_train_augmented)
    X_val_daisy = DAISY.extract_features(X_val)
    end = timer()
    print(f"Time to extract DAISY features: {timedelta(seconds=end-start)}")
    print(f"DAISY features extracted: {X_train_daisy.shape[1]}")

    # extract HOG features
    HOG = hog_feature_extractor()
    print("Extracting HOG features")
    start = timer()
    X_train_hog = HOG.extract_features(X_train_augmented)
    X_val_hog = HOG.extract_features(X_val)
    end = timer()
    print(f"Time to extract HOG features: {timedelta(seconds=end-start)}")
    print(f"HOG features extracted: {X_train_hog.shape[1]}")

    # Put all Daisy features of train set as one list of elements
    '''
    daisy_feature_list = []
    print("Preparing data to do KMeans on DAISY features")
    for features in tqdm(X_train_daisy):
        daisy_feature_list = daisy_feature_list + list(features)

    daisy_feature_list = np.array(daisy_feature_list)
    '''

    daisy_features_list = np.ravel(X_train_daisy).reshape(-1,1)
    print(daisy_features_list.shape)
    n_clusters = 500 
    assert daisy_features_list.shape[0] > n_clusters, "Number of features must be greater than the number of clusters"
    # Cluster the DAISY features training the model
    daisy_cluster_model=cluster_daisy_features(daisy_features_list, n_clusters) 
    print(f"Number of clusters over Daisy features: {daisy_cluster_model.n_clusters}")

    #Prepare data for training and validation
    X_train = []
    print(f"Preparing features for training and validation")
    for daisy_feature, hog_feature in tqdm(zip(X_train_daisy, X_train_hog), total=len(X_train_daisy)):
        X_train.append(prepare_features(daisy_feature, hog_feature, daisy_cluster_model))
    
    X_val = []
    for daisy_feature, hog_feature in tqdm(zip(X_val_daisy, X_val_hog), total=len(X_val_daisy)):
        X_val.append(prepare_features(daisy_feature, hog_feature, daisy_cluster_model))

    #############TRAIN THE MODEL################
    #Define the kernel and the classifier
    kernel = RBF(sigma=1.6)
    multiKSVC = MulticlassKernelSVC(C=1, kernel=kernel)

    multiKSVC.fit(X_train_hog, Y_train_augmented)

    #Predict the validation set
    y_pred = multiKSVC.predict(X_val_hog)
    print(f"Accuracy on validation set: {np.mean(y_pred == Y_val)}")
    
    ############################################

    #Prepare test data
    Xte_daisy = DAISY.extract_features(Xte)
    Xte_hog = HOG.extract_features(Xte)

    X_test = []
    for daisy_feature, hog_feature in zip(Xte_daisy, Xte_hog):
        X_test.append(prepare_features(daisy_feature, hog_feature, daisy_cluster_model))

    ############TEST THE MODEL##################
    Yte = multiKSVC.predict(Xte_hog) 

    Yte = {'Prediction' : Yte} 
    dataframe = pd.DataFrame(Yte) 
    dataframe.index += 1 
    dataframe.to_csv('./Yte_pred_hog_daisy.csv',index_label='Id')
    





