#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 00:37:06 2018

@author: abhijay
"""

# Comment
import os
if not os.path.exists('Q13_plots'):
    os.mkdir('Q13_plots')
    
#os.getcwd()
#os.chdir("/home/abhijay/Documents/ML/hw_1/11632196/")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
import time

class BinaryClassifier():
    def __init__(self, train_data, dev_data, test_data):
        
        #### Converting integer continuous variables to categories ####
        train_data['age'] = train_data['age'].apply(lambda x: '0-19' if x>0 and x<20 else ( '20-35' if x>=20 and x<35 else( '35-50' if x>=35 and x<50 else '>50' )))
        dev_data['age'] = dev_data['age'].apply(lambda x: '0-19' if x>0 and x<20 else ( '20-35' if x>=20 and x<35 else( '35-50' if x>=35 and x<50 else '>50' )))
        test_data['age'] = test_data['age'].apply(lambda x: '0-19' if x>0 and x<20 else ( '20-35' if x>=20 and x<35 else( '35-50' if x>=35 and x<50 else '>50' )))
        
        train_data['hours-per-week'] = train_data['hours-per-week'].apply(lambda x: '0-15' if x>0 and x<15 else ( '15-30' if x>=15 and x<30 else( '30-45' if x>=30 and x<45 else ( '45-60' if x>=45 and x<60 else '>60' ) )))
        dev_data['hours-per-week'] = dev_data['hours-per-week'].apply(lambda x: '0-15' if x>0 and x<15 else ( '15-30' if x>=15 and x<30 else( '30-45' if x>=30 and x<45 else ( '45-60' if x>=45 and x<60 else '>60' ) )))
        test_data['hours-per-week'] = test_data['hours-per-week'].apply(lambda x: '0-15' if x>0 and x<15 else ( '15-30' if x>=15 and x<30 else( '30-45' if x>=30 and x<45 else ( '45-60' if x>=45 and x<60 else '>60' ) )))
        ####
        
        #### Separating out features from target variable ####
        train_x, self.train_y = self.separate_target_variable(train_data)
        dev_x, self.dev_y = self.separate_target_variable(dev_data)
        test_x, self.test_y = self.separate_target_variable(test_data)
        ####
        
        #### One hot encode all categorical variables ####        
        categorical_features = [train_x.columns[col] for col, col_type in enumerate(train_x.dtypes) if col_type == np.dtype('O') ]
        
        train_x = self.one_hot_encode(train_x, categorical_features)
        dev_x = self.one_hot_encode(dev_x, categorical_features)
        test_x = self.one_hot_encode(test_x, categorical_features)
        ####
        
        # Make features in dev categories consistent with train
        dev_x = self.make_features_correspond( train_x, dev_x)
        test_x = self.make_features_correspond( train_x, test_x)
        
#        train_features = set(train_x.columns[(train_x.var(axis=0)>0.1)].tolist())
#        #dev_features = set(dev_x.columns[(dev_x.var(axis=0)>0.1)].tolist())
#        test_features = set(test_x.columns[(test_x.var(axis=0)>0.1)].tolist())
#        
#        final_list_features = list(train_features.intersection(test_features))
        
        final_list_features = list(train_x.columns)
        
        final_list_features.sort()
        
        train_x = train_x[final_list_features]
        dev_x = dev_x[final_list_features]
        test_x = test_x[final_list_features]
        
        # Now that the features are consistent I can convert my datasets into numpy arrays
        
        #### Now that the features are consistent, convert datasets into numpy arrays ####
        train_x = np.array(train_x.values)
        dev_x = np.array(dev_x.values)
        test_x = np.array(test_x.values)
                
#        scaler = StandardScaler().fit(train_x)
#        self.train_x = scaler.transform(train_x)
#        self.test_x = scaler.transform(test_x)
#        self.dev_x = scaler.transform(dev_x)
        
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(train_x.astype(float))
        self.train_x = scaling.transform(train_x)
        self.dev_x = scaling.transform(dev_x)
        self.test_x = scaling.transform(test_x)
        ####

    def perceptron(self, iterations, train_x, train_y):
        
        mistakes_list = []
        train_accuracy = []
        dev_accuracy = []
        test_accuracy = []
        
        w = np.zeros( train_x.shape[1])
        tau = 1
        
        for itr in range(iterations):
            mistake_count = 0
            for x,y in zip( train_x, train_y):
                y_hat = np.sign(np.dot(w,x))
                if y_hat != y:
                    mistake_count+=1
                    w = w + tau * y * x
            
            mistakes_list.append( mistake_count)
            train_accuracy.append(self.predict(w, train_x, train_y))
            dev_accuracy.append(self.predict(w, self.dev_x, self.dev_y))
            test_accuracy.append(self.predict(w, self.test_x, self.test_y))
        
        return mistakes_list, train_accuracy, dev_accuracy, test_accuracy

    def passive_aggresive_perceptron(self, iterations, train_x, train_y):
        
        mistakes_list = []
        train_accuracy = []
        dev_accuracy = []
        test_accuracy = []
        
        w = np.zeros( train_x.shape[1])
    
        for itr in range(iterations):
            mistake_count = 0
            for x,y in zip( train_x, train_y):
                y_hat = np.sign(np.dot(w,x))
                if y_hat != y:
                    mistake_count+=1
                    tau = float(1 - (y*np.dot(w,x)) )/np.linalg.norm( x, ord=1)**2
                    w = w + tau * y * x
            
            mistakes_list.append( mistake_count)
            train_accuracy.append(self.predict(w, train_x, train_y))
            dev_accuracy.append(self.predict(w, self.dev_x, self.dev_y))
            test_accuracy.append(self.predict(w, self.test_x, self.test_y))
        
        return mistakes_list, train_accuracy, dev_accuracy, test_accuracy
    
    def naive_average_perceptron(self, iterations):
        
        # IMP IMP IMP   Measure the training time to perform 5 iterations for both implementations. 
        
        train_accuracy = []
        dev_accuracy = []
        test_accuracy = []
        
        w = np.zeros(self.train_x.shape[1])
        w_sum = np.zeros(self.train_x.shape[1])
        learning_rate = 1
        
        count = 0
        for itr in range(iterations):
            for x,y in zip( self.train_x, self.train_y):
                y_hat = np.sign(np.dot(w,x))
                if y_hat != y:                
                    w = w + learning_rate * y * x
                    w_sum += w
                    count+=1
            w_avg = w_sum/count
            # print (w_avg)
            
            train_accuracy.append(self.predict(w_avg, self.train_x, self.train_y))
            dev_accuracy.append(self.predict(w_avg, self.dev_x, self.dev_y))
            test_accuracy.append(self.predict(w_avg, self.test_x, self.test_y))
                    
        return train_accuracy, dev_accuracy, test_accuracy
    
    def smart_average_perceptron(self, iterations):
        
        train_accuracy = []
        dev_accuracy = []
        test_accuracy = []
        
        w = np.zeros(self.train_x.shape[1])
        u = np.zeros(self.train_x.shape[1])
        b = 0
        beta = 0
        c = 1
    
        for itr in range(iterations):
            for x,y in zip( self.train_x, self.train_y):
                if y*(np.dot(w,x)+b)<=0:                
                    w = w + y*x
                    b = b + y
                    u = u+ y*c*x
                    beta = beta + y*c
                c = c + 1
            
            train_accuracy.append(self.predict( (w-(u/c)), self.train_x, self.train_y, (b-(beta/c))))
            dev_accuracy.append(self.predict( (w-(u/c)), self.dev_x, self.dev_y, (b-(beta/c))))
            test_accuracy.append(self.predict( (w-(u/c)), self.test_x, self.test_y, (b-(beta/c))))
                    
        return train_accuracy, dev_accuracy, test_accuracy
    
    def compareTrainingTime(self, iterations):
        
        # Naive average perceptron
        start_time = time.time()
        w = np.zeros(self.train_x.shape[1])
        w_sum = np.zeros(self.train_x.shape[1])
        learning_rate = 1
        
        count = 0
        for itr in range(iterations):
            for x,y in zip( self.train_x, self.train_y):
                y_hat = np.sign(np.dot(w,x))
                if y_hat != y:                
                    w = w + learning_rate * y * x
                    w_sum += w
                    count+=1
            w_avg = w_sum/count
        print("--- Naive avg perceptron took: %s seconds ---" % (time.time() - start_time))        
        
        # Smart average perceptron
        start_time = time.time()
        w = np.zeros(self.train_x.shape[1])
        u = np.zeros(self.train_x.shape[1])
        b = 0
        beta = 0
        c = 1
    
        for itr in range(iterations):
            for x,y in zip( self.train_x, self.train_y):
                if y*(np.dot(w,x)+b)<=0:                
                    w = w + y*x
                    b = b + y
                    u = u+ y*c*x
                    beta = beta + y*c
                c = c + 1
        print("--- Smart avg perceptron took: %s seconds ---" % (time.time() - start_time))

    def plot_figure( self, blueLineValues, blueLineLabel, orangeLineValues, orangeLineLabel, y_limit, title, y_label, x_label, legendLoc, saveAs):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot( range(1,len(blueLineValues)+1), blueLineValues, label = blueLineLabel)
        ax.plot( range(1,len(orangeLineValues)+1), orangeLineValues, label = orangeLineLabel)
        ax.set_ylim(y_limit[0], y_limit[1])
        ax.set_title( title, fontsize=18)
        ax.set_ylabel( y_label, fontsize=15)
        ax.set_xlabel( x_label, fontsize=15)
        ax.set_xticks( range(1,len(blueLineValues)+1))
        ax.legend(loc=legendLoc)
        fig.savefig("Q13_plots/"+saveAs)
    
    def plot_figure_forIncremental( self, blueLineValues, blueLineLabel, orangeLineValues, orangeLineLabel, greenLineValues, greenLineLabel, redLineValues, redLineLabel, y_limit, title, y_label, x_label, legendLoc, saveAs):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot( range(1,len(blueLineValues)+1), blueLineValues, label = blueLineLabel)
        ax.plot( range(1,len(orangeLineValues)+1), orangeLineValues, label = orangeLineLabel)
        ax.plot( range(1,len(greenLineValues)+1), greenLineValues, label = greenLineLabel)
        ax.plot( range(1,len(redLineValues)+1), redLineValues, label = redLineLabel)
        ax.set_ylim(y_limit[0]-5, y_limit[1]+5)
        ax.set_title( title, fontsize=18)
        ax.set_ylabel( y_label, fontsize=15)
        ax.set_xlabel( x_label, fontsize=15)
        ax.set_xticks( range(1,len(blueLineValues)+1))
        ax.set_xticklabels( [i*5000 for i in range(1,len(blueLineValues)+1)])
        ax.legend(loc=legendLoc)
        fig.savefig("Q13_plots/"+saveAs)
    
    def separate_target_variable(self, data):
        data_x = data.iloc[:,0:-1]
        data_y = data.iloc[:,-1]
        data_y = data_y.apply(lambda y: -1 if y==" <=50K" else 1).tolist()
        return data_x, data_y

    def make_features_correspond(self, train_, data_):
        missing_features = set(train_.columns) - set(data_.columns)
        missing_features = pd.DataFrame(0, index=np.arange(len(data_)), columns=missing_features)
        data_ = pd.concat( [data_, missing_features], axis = 1)
        return data_ 
    
    def one_hot_encode(self, data, categorical_features):    
        for categorical_feature in categorical_features:
            dummies = pd.get_dummies(data[categorical_feature],prefix=categorical_feature)
            dummies = dummies.iloc[:,0:-1]
            data = pd.concat( [data, dummies], axis = 1).drop(categorical_feature, axis=1) # Also add logic to remove redundant info
        return data
    
    def predict( self, w, x, y, b=0):
        if b == 0:
            b = np.zeros_like(np.dot(w,x.T))
            
        y_hat = np.sign(np.dot(w,x.T)+b)
        correct = np.sum((y_hat == np.array(y)).astype(int))
        # print (correct, x.shape[0])
        # print (round( (float(correct)/x.shape[0])*100 , 3))
        return round( (float(correct)/x.shape[0])*100 , 3)

if __name__ == "__main__":
    
    print ("\n\n =============== Perceptron ===============\n")
    
    # Get data
    col_names = ["age","workclass","education","marital_status","occupation","race","gender","hours-per-week","native-country","salary-bracket"]
    
    train_data = pd.read_csv("income-data/income.train.txt", names = col_names)
    dev_data = pd.read_csv("income-data/income.dev.txt", names = col_names)
    test_data = pd.read_csv("income-data/income.test.txt", names = col_names)
    
    print ("\nPreparing data.....\n")
    binary_classifier = BinaryClassifier( train_data, dev_data, test_data)
    
    print ("\n\n =============== Standard Perceptron and Passive Aggressive Perceptron ===============\n")
    
    print ("\nTraining and Testing on Standard Perceptron and Passive Aggressive Perceptron.....\n")
    ##### Standard Perceptron vs Passive Aggressive Perceptron #####
    perceptron_train_mistakes, perceptron_train_accuracy, perceptron_dev_accuracy, perceptron_test_accuracy = binary_classifier.perceptron(5, binary_classifier.train_x, binary_classifier.train_y)
    pa_perceptron_train_mistakes, pa_perceptron_train_accuracy, pa_perceptron_dev_accuracy, pa_perceptron_test_accuracy = binary_classifier.passive_aggresive_perceptron(5, binary_classifier.train_x, binary_classifier.train_y)
    
    print ("\nPlotting for Standard and PA perceptron\n")
    binary_classifier.plot_figure( perceptron_train_mistakes, 'perceptron', pa_perceptron_train_mistakes, 'pa_perceptron', (min(perceptron_train_mistakes+pa_perceptron_train_mistakes)-200, max(perceptron_train_mistakes+pa_perceptron_train_mistakes)+200), 'Standard vs Passive Aggressive - Mistakes while training', 'Mistakes', 'Iteration', 'upper right', 'Standard vs Passive Aggressive - Mistakes_while_training.png')
    binary_classifier.plot_figure( perceptron_train_accuracy, 'perceptron', pa_perceptron_train_accuracy, 'pa_perceptron', ( min(perceptron_train_accuracy+pa_perceptron_train_accuracy)-5, max(perceptron_train_accuracy+pa_perceptron_train_accuracy)+5), 'Standard vs Passive Aggressive - Accuracy on training data', 'Accuracy', 'Iteration', 'lower right', 'Standard vs Passive Aggressive - Accuracy_on_training_data.png')
    binary_classifier.plot_figure( perceptron_dev_accuracy, 'perceptron', pa_perceptron_dev_accuracy, 'pa_perceptron', (min(perceptron_dev_accuracy+pa_perceptron_dev_accuracy)-5, max(perceptron_dev_accuracy+pa_perceptron_dev_accuracy)+5), 'Standard vs Passive Aggressive - Accuracy on dev data', 'Accuracy', 'Iteration', 'lower right', 'Standard vs Passive Aggressive - Accuracy_on_dev_data.png')
    binary_classifier.plot_figure( perceptron_test_accuracy, 'perceptron', pa_perceptron_test_accuracy, 'pa_perceptron', (min(perceptron_test_accuracy+pa_perceptron_test_accuracy)-5, max(perceptron_test_accuracy+pa_perceptron_test_accuracy)+5), 'Standard vs Passive Aggressive - Accuracy on test data', 'Accuracy', 'Iteration', 'lower right', 'Standard vs Passive Aggressive - Accuracy_on_test_data.png')
    
    #####
    print ("\nVarying data in increments of 5000 and training\n")
    perceptron_train_mistakes, perceptron_train_accuracy, perceptron_dev_accuracy_5000, perceptron_test_accuracy_5000 = binary_classifier.perceptron(5, binary_classifier.train_x[0:5000], binary_classifier.train_y[0:5000])
    pa_perceptron_train_mistakes, pa_perceptron_train_accuracy, pa_perceptron_dev_accuracy_5000, pa_perceptron_test_accuracy_5000 = binary_classifier.passive_aggresive_perceptron(5, binary_classifier.train_x[0:5000], binary_classifier.train_y[0:5000])

    perceptron_train_mistakes, perceptron_train_accuracy, perceptron_dev_accuracy_10000, perceptron_test_accuracy_10000 = binary_classifier.perceptron(5, binary_classifier.train_x[0:10000], binary_classifier.train_y[0:10000])
    pa_perceptron_train_mistakes, pa_perceptron_train_accuracy, pa_perceptron_dev_accuracy_10000, pa_perceptron_test_accuracy_10000 = binary_classifier.passive_aggresive_perceptron(5, binary_classifier.train_x[0:10000], binary_classifier.train_y[0:10000])

    perceptron_train_mistakes, perceptron_train_accuracy, perceptron_dev_accuracy_15000, perceptron_test_accuracy_15000 = binary_classifier.perceptron(5, binary_classifier.train_x[0:15000], binary_classifier.train_y[0:15000])
    pa_perceptron_train_mistakes, pa_perceptron_train_accuracy, pa_perceptron_dev_accuracy_15000, pa_perceptron_test_accuracy_15000 = binary_classifier.passive_aggresive_perceptron(5, binary_classifier.train_x[0:15000], binary_classifier.train_y[0:15000])

    perceptron_train_mistakes, perceptron_train_accuracy, perceptron_dev_accuracy_20000, perceptron_test_accuracy_20000 = binary_classifier.perceptron(5, binary_classifier.train_x[0:20000], binary_classifier.train_y[0:20000])
    pa_perceptron_train_mistakes, pa_perceptron_train_accuracy, pa_perceptron_dev_accuracy_20000, pa_perceptron_test_accuracy_20000 = binary_classifier.passive_aggresive_perceptron(5, binary_classifier.train_x[0:20000], binary_classifier.train_y[0:20000])

    incrementalPerceptronDevAccuracy = [perceptron_dev_accuracy_5000[4], perceptron_dev_accuracy_10000[4], perceptron_dev_accuracy_15000[4], perceptron_dev_accuracy_20000[4], perceptron_dev_accuracy[4]]
    incrementalPAPerceptronDevAccuracy = [pa_perceptron_dev_accuracy_5000[4], pa_perceptron_dev_accuracy_10000[4], pa_perceptron_dev_accuracy_15000[4], pa_perceptron_dev_accuracy_20000[4], pa_perceptron_dev_accuracy[4]]    
    incrementalPerceptronTestAccuracy = [perceptron_test_accuracy_5000[4], perceptron_test_accuracy_10000[4], perceptron_test_accuracy_15000[4], perceptron_test_accuracy_20000[4], perceptron_test_accuracy[4]]
    incrementalPAPerceptronTestAccuracy = [pa_perceptron_test_accuracy_5000[4], pa_perceptron_test_accuracy_10000[4], pa_perceptron_test_accuracy_15000[4], pa_perceptron_test_accuracy_20000[4], pa_perceptron_test_accuracy[4]]
    
    y_limit = incrementalPerceptronDevAccuracy+incrementalPAPerceptronDevAccuracy+incrementalPerceptronTestAccuracy+incrementalPAPerceptronTestAccuracy
    binary_classifier.plot_figure_forIncremental( incrementalPerceptronDevAccuracy, "Perceptron Dev Accuracy", incrementalPAPerceptronDevAccuracy, "PA Perceptron Dev Accuracy", incrementalPerceptronTestAccuracy, "Perceptron Test Accuracy", incrementalPAPerceptronTestAccuracy, "PA Perceptron Test Accuracy", [min(y_limit),max(y_limit)], "Accuracy plots - Varying number of samples for Training", "Accuracy", "Number of Samples", "lower right", "Accuracy plots - Varying number of samples for Training")
    #####
    
    print ("\n\n =============== Average Perceptron ===============\n")
    print ("\nTraining and Testing on Naive Average Perceptron vs Smart Average Perceptron.....\n")
    ##### Naive Average Perceptron vs Smart Average Perceptron #####
    
    naive_avg_perceptron_train_accuracy, naive_avg_perceptron_dev_accuracy, naive_avg_perceptron_test_accuracy = binary_classifier.naive_average_perceptron(5)    
    smart_avg_perceptron_train_accuracy, smart_avg_perceptron_dev_accuracy, smart_avg_perceptron_test_accuracy = binary_classifier.smart_average_perceptron(5)
       
    binary_classifier.plot_figure( naive_avg_perceptron_train_accuracy, 'Naive avg perceptron', smart_avg_perceptron_train_accuracy, 'Smart avg perceptron', (83, 84), 'Accuracy on training data Average Perceptron', 'Accuracy', 'Iteration', 'upper right', 'AvgPerceptron_Accuracy_on_training_data.png')
    binary_classifier.plot_figure( naive_avg_perceptron_dev_accuracy, 'Naive avg perceptron', smart_avg_perceptron_dev_accuracy, 'Smart avg perceptron', (82, 84), 'Accuracy on dev data Average Perceptron', 'Accuracy', 'Iteration', 'upper right', 'AvgPerceptron_Accuracy_on_dev_data.png')
    binary_classifier.plot_figure( naive_avg_perceptron_test_accuracy, 'Naive avg perceptron', smart_avg_perceptron_test_accuracy, 'Smart avg perceptron', (82, 84), 'Accuracy on test data Average Perceptron', 'Accuracy', 'Iteration', 'upper right', 'AvgPerceptron_Accuracy_on_test_data.png')

    binary_classifier.compareTrainingTime(5)
    
    print ("Ans 13 Done.....")
    print ("\n==============================\n\n")
    #####   
    