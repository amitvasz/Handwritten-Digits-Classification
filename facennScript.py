'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import math
import pickle

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return expit(z)

def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label_matrix, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0.0
    obj_sum = 0.0
    no_of_inputs = training_data.shape[0]
    bias = np.ones(no_of_inputs)
    training_data = np.insert(training_data,0,bias,axis=1)
    sig = np.dot(w1,np.transpose(training_data))
    outputOfHiddenLayer = sigmoid(sig)
    outputOfHiddenLayer = np.insert(outputOfHiddenLayer,0,bias,axis=0)

    sig_for_final_output = np.dot(w2,outputOfHiddenLayer)
    finalOutput = np.transpose(sigmoid(sig_for_final_output))
    
    '''training_label_matrix = np.ndarray(shape=(no_of_inputs,n_class))
    
    print("Shape of training label",np.shape(training_label))
    for i in range(len(training_label)):
        for j in range(n_class):
            if(j == training_label[i]):
                training_label_matrix[i][j] = 1
            
            else:
                training_label_matrix[i][j] = 0   '''
    
                
    
    
    #obj_val = np.sum(training_label_matrix*np.log(finalOutput) + (1-training_label_matrix)*np.log(1-finalOutput))
    
    #obj_val = (obj_val * -1)/no_of_inputs
    for i in range(no_of_inputs):
        intermediateSum = 0.0
        for j in range(n_class):
            if j == training_label_matrix[i][j]:
                    intermediateSum = intermediateSum + np.log(finalOutput[i][j])
                
            else:
                    diff = np.log(1-finalOutput[i][j])
                    intermediateSum = intermediateSum + diff

        obj_sum = (obj_sum + intermediateSum)
   
    obj_val = (obj_sum * -1)/no_of_inputs
    
    print("Objective Value is : ",obj_val)   
    #print(np.shape(training_label_matrix))
    gradiance = np.subtract(finalOutput, training_label_matrix)
    #print("Shape of O-L",np.shape(gradiance))
    grad_w2 = np.dot(np.transpose(gradiance),np.transpose(outputOfHiddenLayer))
    #print(np.shape(grad_w2))            
    
    #Calculating Obj Function
            
    subtraction = 1-outputOfHiddenLayer
    #print("Subtraction Shape",np.shape(subtraction))
    sub_prod_zJ = subtraction*outputOfHiddenLayer
    #print("sub_prod_zJ",np.shape(sub_prod_zJ))
    output_of_summation = np.dot(gradiance,w2)
    #print("output of summation Shape",np.shape(output_of_summation))
    prod_summation = sub_prod_zJ*np.transpose(output_of_summation)
    prod_summation = np.delete(prod_summation,0,axis=0)
    #print("prod_summation Shape",np.shape(prod_summation))
    grad_w1 = np.dot(prod_summation,training_data)
    #print("Shape of Grad W1",np.shape(grad_w1))
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    
    regularization_of_w1 = np.sum(np.square(w1))
    regularization_of_w2 = np.sum(np.square(w2))
    regularization = (lambdaval*(regularization_of_w1+regularization_of_w2))/(2*no_of_inputs)
    obj_val = obj_val + regularization
        
    delJDividedDwj2 = (grad_w2 + lambdaval*w2)/(no_of_inputs)
    delJDividedDwj1 = (grad_w1 + lambdaval*w1)/(no_of_inputs)
    
    
    obj_grad = np.array([])
    obj_grad = np.concatenate((delJDividedDwj1.flatten(), delJDividedDwj2.flatten()),0)
    
    return (obj_val, obj_grad)
    
def nnPredict(w1,w2,data):
    no_of_input = data.shape[0]
    bias = np.ones(no_of_input)
    data = np.insert(data,0,bias,axis=1)
    sig = np.dot(w1,np.transpose(data))
    outputOfHiddenLayer = sigmoid(sig)
    outputOfHiddenLayer = np.insert(outputOfHiddenLayer,0,bias,axis=0)
    #print("Output of Hidden Layer\n")
    #print(outputOfHiddenLayer)
    
    sig = np.dot(w2,outputOfHiddenLayer)
    finalOutput = np.transpose(sigmoid(sig))
    #print("Output of Final Layer\n")
    #print(np.shape(finalOutput))

    labels = np.ndarray(shape=(no_of_input,n_class))
    for i in range(len(finalOutput)):
        maxValue = max(finalOutput[i,:])
        flag = 0
        for j in range(finalOutput.shape[1]):
            if maxValue == finalOutput[i][j] and flag != 1:
                labels[i][j] = 1
                flag = 1
            
            else:
                labels[i][j] = 0
    # Your code here
    print(np.shape(labels))        
    return labels    

def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = np.zeros(shape=(21100, 2))
    train_l = labels[0:21100]
    valid_y = np.zeros(shape=(2665, 2))
    valid_l = labels[21100:23765]
    test_y = np.zeros(shape=(2642, 2))
    test_l = labels[23765:]
    for i in range(train_y.shape[0]):
        train_y[i, train_l[i]] = 1
    for i in range(valid_y.shape[0]):
        valid_y[i, valid_l[i]] = 1
    for i in range(test_y.shape[0]):
        test_y[i, test_l[i]] = 1

    return train_x, train_y, valid_x, valid_y, test_x, test_y
"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 60
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
