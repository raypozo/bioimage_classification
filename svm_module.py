import numpy as np
import random
import math
import matplotlib.pyplot as plt
from numpy import *
from sklearn import svm
# Function to select a set of unique random points in a defined range
def random_query(n,idx,num_queries):
    for i in range(num_queries):
        new_point=random.choice([x for x in range(n) if x not in idx])
        idx.append(new_point)
    return idx

def svm_model(X,Y,kernel_fun):
    model=svm.SVC(probability=True,kernel=kernel_fun)
    model.fit(X,Y)

    return model

def svm_random(X_l,X_u,Y_l,Y_u,batch_size):
    n=X_u.shape[0]
    queried_idx=random_query(n,[],batch_size)
    X_l=np.concatenate((X_l,X_u[queried_idx][:]))
    X_u=np.delete(X_u,queried_idx,0)
    Y_l=np.concatenate((Y_l,Y_u[0][queried_idx]))
    Y_u=np.delete(Y_u,queried_idx,1)
    return X_l,X_u,Y_l,Y_u

#Query by Committee implementation
#Function that selects batch of points which have the highest Hard Vote Entropy
#svm_settings contain different kernel functions, one for each committee member
def svm_query_by_committee(X_l,X_u,Y_l,Y_u,batch_size):
    svm_settings=['linear','rbf','sigmoid']
    c_size=len(svm_settings)   #This is the size of the Committee
    u_size=X_u.shape[0]
    entropies=np.zeros([1,u_size],dtype=float)
    #Construct different committees with different settings
    models=[]
    for i in range(c_size):
        model=svm_model(X_l,Y_l,svm_settings[i])
        models.append(model)
    pred_models=[]
    #Apply each model to every point in the unlabeled pool
    for i in range(c_size):
        pred_model=models[i].predict(X_u)
        pred_models.append(pred_model)
    for i in range(len(pred_models[0])):
        label_counts=np.zeros([1,8],dtype=float)
        for j in range(c_size):
            idx=pred_models[j][i]-1
            label_counts[0,idx]+=1
        label_counts=np.divide(label_counts,c_size)
        log_label_counts=ma.log(label_counts).filled(0)
        entropy=abs(np.sum(label_counts*log_label_counts))
        entropies[0,i]=entropy
    # Find the indices of the points with highest entropy
    top_uncertain=entropies.argsort()[0,-batch_size:]
    X_l=np.concatenate((X_l,X_u[top_uncertain][:]))
    X_u=np.delete(X_u,top_uncertain,0)
    Y_l=np.concatenate((Y_l,Y_u[0][top_uncertain]))
    Y_u=np.delete(Y_u,top_uncertain,1)

    return X_l,X_u,Y_l,Y_u

def svm_uncertainty_sampling(X_l,X_u,Y_l,Y_u,batch_size):
    model=svm_model(X_l,Y_l,'rbf')
    prob=model.predict_proba(X_u)
    #Sort Probabilities by row
    prob.sort()
    #Get Largest probability per row
    last_column=prob[:,-1]
    #Sort instances which have the smallest probability(most uncertain)
    idx=np.argsort(last_column)
    top_uncertain=idx[:batch_size]
    #Add queried points to the Labeled Pool and remove from Unlabeled Pool
    X_l=np.concatenate((X_l,X_u[top_uncertain][:]))
    X_u=np.delete(X_u,top_uncertain,0)
    Y_l=np.concatenate((Y_l,Y_u[0][top_uncertain]))
    Y_u=np.delete(Y_u,top_uncertain,1)
    return X_l,X_u,Y_l,Y_u

#Function to plot the
def plot_svm(rdm_accuracy,active_accuracy,rdm_error,active_error,cost,active_method,data_pool):
    enum={1:'EASY',2:'MODERATE',3:'DIFFICULT'}
    enum2={1:'Uncertainty Sampling',2:'Query by Committee'}
    plt.figure(1)
    plt.plot(cost,rdm_error)
    plt.plot(cost,active_error)
    plt.legend(['Random Learner','Active Learner'])
    plt.ylabel('Error')
    plt.xlabel('Cost')
    plt.title('SVM with %s, Dataset: %s (Error)'%(enum2[active_method],enum[data_pool]))
    plt.figure(2)
    plt.plot(cost,rdm_accuracy)
    plt.plot(cost,active_accuracy)
    plt.legend(['Random Learner','Active Learner'])
    plt.ylabel('Accuracy')
    plt.xlabel('Cost')
    plt.title('SVM with %s, Dataset: %s (Accuracy)'%(enum2[active_method],enum[data_pool]))
    plt.show()
    return

def run_svm(train_data,train_labels,test_data,test_labels,budget,active_method,data_pool):

    init_size=50  #Initial number of Queries to populate Labeled Pool
    batch_size=50
    n=train_data.shape[0]
    queried_idx=[]
    queried_idx=random_query(n,queried_idx,init_size)
    print('Training Algorithm: Could take a few minutes...')
    #Active Model initialization
    active_l_data=train_data[queried_idx][:]
    active_u_data=np.delete(train_data,queried_idx,0)
    active_l_labels=train_labels[0][queried_idx]
    active_u_labels=np.delete(train_labels,queried_idx,1)
    #Random model initialization
    rdm_l_data=train_data[queried_idx][:]
    rdm_u_data=np.delete(train_data,queried_idx,0)
    rdm_l_labels=train_labels[0][queried_idx]
    rdm_u_labels=np.delete(train_labels,queried_idx,1)
    budget-=init_size
    total_cost=init_size
    rdm_accuracy=[]
    active_accuracy=[]
    cost=[]
    rdm_error=[]
    active_error=[]

    for i in range(budget/batch_size):
        if active_method==1:
            active_l_data,active_u_data,active_l_labels,active_u_labels=svm_uncertainty_sampling(active_l_data,active_u_data,active_l_labels,active_u_labels,batch_size)
        elif active_method==2:
            active_l_data,active_u_data,active_l_labels,active_u_labels=svm_query_by_committee(active_l_data,active_u_data,active_l_labels,active_u_labels,batch_size)
        else:
            print("Not valid Active Method Selection")
            return

        rdm_l_data,rdm_u_data,rdm_l_labels,rdm_u_labels=svm_random(rdm_l_data,rdm_u_data,rdm_l_labels,rdm_u_labels,batch_size)
        #Train Active-learning and Random model independently
        model_rdm=svm_model(rdm_l_data,rdm_l_labels,'rbf')
        model_active=svm_model(active_l_data,active_l_labels,'rbf')
        #Make predictions with each model over the test dataset
        pred_rdm=model_rdm.predict(test_data)
        pred_active=model_active.predict(test_data)
        size_pred=pred_rdm.shape[0]
        #Calculate Accuracy over the test set per iteration and store
        correct_pred_rdm=(size_pred-np.count_nonzero(np.abs(pred_rdm-test_labels)))/float(size_pred)
        error_pred_rdm=(np.count_nonzero(np.abs(pred_rdm-test_labels)))/float(size_pred)
        correct_pred_active=(size_pred-np.count_nonzero(np.abs(pred_active-test_labels)))/float(size_pred)
        error_pred_active=(np.count_nonzero(np.abs(pred_active-test_labels)))/float(size_pred)
        rdm_accuracy.append(correct_pred_rdm)
        rdm_error.append(error_pred_rdm)
        active_accuracy.append(correct_pred_active)
        active_error.append(error_pred_active)
        total_cost+=batch_size
        cost.append(total_cost)
    #Plot Results
    print()
    print('Ploting results...')
    plot_svm(rdm_accuracy,active_accuracy,rdm_error,active_error,cost,active_method,data_pool)

    return model_active
