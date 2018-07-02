import argparse
import numpy as np
import os,sys
import random
import math
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from numpy import *
from sklearn import svm
import knn_module
import svm_module

#Function to parse the user input
def create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_pool', type=int, default=1,
                      help='Data Pool, select integer value for option: 1)easy, 2)moderate, 3)difficult')
  parser.add_argument('--active_method', type=int, default=1,
                      help='Active Learning method, select integer value for option: 1)Uncertainty Sampling, 2) Query by Committee')
  parser.add_argument('--algorithm', type=int, default=1,
                      help='Supported algorithms: 1)KNN, 2)SVM')
  return parser

# Function to Load Training and Testing data
def read_data(data_pool):
    patterns={'Endosomes':1,'Lysosome':2,'Mitochondria':3,'Peroxisomes':4,'Actin':5,'Plasma_Membrane':6, 'Microtubules':7, 'Endoplasmic_Reticulum':8}
    if data_pool==1:
        train_file = 'EASY_TRAIN.csv'
        test_file = 'EASY_TEST.csv'
        num_features=26
    elif data_pool==2:
        train_file = 'MODERATE_TRAIN.csv'
        test_file = 'MODERATE_TEST.csv'
        num_features=26
    elif data_pool==3:
        train_file = 'DIFFICULT_TRAIN.csv'
        test_file = 'DIFFICULT_TEST.csv'
        num_features=52
    train_size=4120
    test_size=1000
    #Load Train Data
    file_data=open(train_file,'r')
    train=file_data.read().strip().split('\n')
    file_data.close()
    train_data=np.zeros([train_size,num_features],dtype=float)
    train_labels=np.zeros([1,train_size],dtype=int)
    for i in range(len(train)):
        line=train[i]
        l=line.strip().split(',')
        train_data[i][:]=l[:-1]
        train_labels[0][i]=patterns[l[-1]]
    #Load Test Data
    file_data=open(test_file,'r')
    test=file_data.read().strip().split('\n')
    file_data.close()
    test_data=np.zeros([test_size,num_features],dtype=float)
    test_labels=np.zeros([1,test_size],dtype=int)

    for i in range(len(test)):
        line=test[i]
        l=line.strip().split(',')
        test_data[i][:]=l[:-1]
        test_labels[0][i]=patterns[l[-1]]

    return train_data,train_labels,test_data,test_labels

# Function to Load Blind Dataset
def read_blind(data_pool):
    if data_pool==1:
        blind_file = 'EASY_BLINDED.csv'
        num_features=26
    elif data_pool==2:
        blind_file = 'MODERATE_BLINDED.csv'
        num_features=26
    elif data_pool==3:
        blind_file = 'DIFFICULT_BLINDED.csv'
        num_features=52
    file_data=open(blind_file,'r')
    blind=file_data.read().strip().split('\n')
    file_data.close()
    blind_size=len(blind)
    test_blinded=np.zeros([blind_size,num_features],dtype=float)
    test_ids=[]
    for i in range(blind_size):
        line=blind[i]
        l=line.strip().split(',')
        test_blinded[i][:]=l[1:]
        test_ids.append(l[0])
    return test_blinded, test_ids

#Function to store Blinded predictions
def store_blinded_predictions(data_pool,test_ids,predictions):
    enum={1:'EASY',2:'MODERATE',3:'DIFFICULT'}
    dec_pat={1:'Endosomes',2:'Lysosome',3:'Mitochondria',4:'Peroxisomes',5:'Actin',6:'Plasma_Membrane',7:'Microtubules',8:'Endoplasmic_Reticulum'}
    b_file = open('%s_BLINDED_PRED.csv'%enum[data_pool], 'w')
    for i in range(len(test_ids)):
        idx=test_ids[i]
        pred=predictions[i]
        b_file.write("%s,%s\n" %(str(idx),dec_pat[pred]))
    b_file.close()
    return

if __name__=='__main__':
    args = create_parser().parse_args()
    data_pool=args.data_pool
    budget=2500
    active_method=args.active_method
    train_data,train_labels,test_data,test_labels=read_data(data_pool)
    if args.algorithm==1:
        model=knn_module.run_knn(train_data,train_labels,test_data,test_labels,budget,active_method,data_pool)
        print('Predicting Blinded dataset...')
        test_blinded,test_ids=read_blind(data_pool)
        predictions=model.predict(test_blinded)
        print('Storing Results...')
        store_blinded_predictions(data_pool,test_ids,predictions)
        print('Finish.')
    elif args.algorithm==2:
        model=svm_module.run_svm(train_data,train_labels,test_data,test_labels,budget,active_method,data_pool)
        print('Predicting Blinded dataset...')
        test_blinded,test_ids=read_blind(data_pool)
        predictions=model.predict(test_blinded)
        print('Storing Results...')
        store_blinded_predictions(data_pool,test_ids,predictions)
        print('Finish.')
    else:
        print('Algorithm selection not valid')
