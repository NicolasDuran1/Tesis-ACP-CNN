#!/usr/bin/env python
# coding: utf-8

# In[2]:

#Bibliotecas
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import linalg #SVD
import screed 
import itertools
import matplotlib.pyplot as plt
import random
import statistics
from collections import Counter
import pickle

random_seed = 600
np.random.seed(random_seed)
random.seed(random_seed)

import sys
import os
os.environ['PYTHONHASHSEED'] = '0'

import tensorflow as tf
tf.compat.v1.random.set_random_seed(random_seed)

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.optimizers import Nadam

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#---------------

def print_metrics(metrics, model_name, folder):

    original_stdout = sys.stdout # Save a reference to the original standard output
    db_name = "nf" if int(sys.argv[1]) == 0 else "full"
    filename = 'SavedResults/'+folder+'/'+model_name+'_'+db_name+'.txt'
    with open(filename, 'w') as f:

        sys.stdout = f # Change the standard output to the file we created. 
        db = "No Feature" if int(sys.argv[1]) == 0 else "Full"
        print("Database: ", db," ", model_name," \n")
        print('-'*50)
        if (int(sys.argv[1]) == 0):
            print('Model Selection: Onehot | k-mers Sparse Matrix | Autoencoder')
        elif (int(sys.argv[1]) == 1):
            print('Model Selection: Onehot | k-mers Sparse Matrix | Autoencoder | (AAC - DPC - PCP)')
        print('-'*50)
        print('Accuracy Training: ', statistics.mean(metrics["accuracy_train"])*100)
        print('Accuracy Testing: ', statistics.mean(metrics["accuracy_test"])*100)
        print('Precision: ', statistics.mean(metrics["precision"])*100)
        print('Sensitivity: ', statistics.mean(metrics["sensitivity"])*100) 
        print('Specificity: ', statistics.mean(metrics["specificity"])*100)
        print('f_1 Score: ', statistics.mean(metrics["f1"])*100)
        print('MCC: ', statistics.mean(metrics["mcc"])*100) 
        print('AUC Score: ', statistics.mean(metrics["auc"])*100) 
        print('MSE: ', statistics.mean(metrics["mse"]))
        print('Mis-Classification: ', statistics.mean(metrics["misc"])) 

        #Mostrar más decimales en DF
        pd.set_option("display.precision", 15)

        #-------------------------------------------
        metrics_model = [metrics["accuracy_train"], metrics["accuracy_test"], metrics["precision"], 
                        metrics["sensitivity"], metrics["specificity"], metrics["mcc"], metrics["auc"], 
                        metrics["f1"], metrics["misc"]]
        metrics_m = pd.DataFrame(metrics_model, columns = ['1', '2','3', '4', '5'],
                    index = ['Accuracy Training','Accuracy Test', 'Precision', 'Sensitivity', 
                            'Specificity', 'MCC', 'AUC Score', 'f_1 Score','Mis-Classification'])
        print(metrics_m)
        sys.stdout = original_stdout # Reset the standard output to its original value

def build_datasets(df_model):
    X = df_model.iloc[:, :-1]

    y = df_model.iloc[:,-1]

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X, y

def confusion_metrics(conf_matrix):
    # Guardar la matriz de confusión y dividirla en 4 piezas
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]

    # Calcular Precisión
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))

    # Calcular mis-classification
    conf_misclassification = 1- conf_accuracy

    # Calcular Sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # Calcular Specificity
    conf_specificity = (TN / float(TN + FP))

    # Calcular la Precisión
    conf_precision = (TP / float(TP + FP))
    # Calcular f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))

    precision = conf_precision
    sensitivity = conf_sensitivity
    specificity = conf_specificity

    return precision, sensitivity, specificity, conf_f1, conf_misclassification

def create_model(input_shape, n_outputs):
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = Sequential()
    model.add(Conv1D(filters=100, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=100, kernel_size=4, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=100, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def change_index(df):
    #Cambiar indices de filas y columnas por valores numéricos
    total_rows_df = df.shape[0]
    df.index = np.arange(0, total_rows_df)

    total_columns_df = df.shape[1]
    df.columns = np.arange(0, total_columns_df)

    return df

def model_selection(alphabeth):

    path_alphabet_nf = 'Databases/NoFeature/nf_Polarity.txt'
    path_alphabet_full = 'Databases/Full/full_Polarity.txt'

    df_nf = pd.read_csv(path_alphabet_nf, sep=" ", header=None)
    #ind_nf = pd.read_csv(path_independent_nf, sep=" ", header=None)

    df_full = pd.read_csv(path_alphabet_full, sep=" ", header=None)
    #ind_full = pd.read_csv(path_independent_full, sep=" ", header=None)

    if (int(sys.argv[1]) == 0):
        df_model = df_nf
        db = 'NoFeature'
        #df_ind = ind_nf
    elif (int(sys.argv[1]) == 1):
        df_model = df_full
        db = 'Full'
        #df_ind = ind_full

    #frames = [df_model_temp, df_ind]
    #df_model = pd.concat(frames)

    df_model = change_index(df_model)

    X, y = build_datasets(df_model)

    # 90% Training - 10% Testing
    X_main, X_testing, y_main, y_testing = train_test_split(X, y, test_size=0.1, random_state=42)
    y_main = y_main.to_numpy()

    # Model ---------------------------------------------------------------------------------------
    epochs = 50
    batch_size = 50
    verbose = 0
    count = 1

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    kf.get_n_splits(X_main)

    dict_metrics = {alphabeth : {'accuracy_train':[], 'accuracy_test':[], 'accuracy_test_ext':[], 
                                'precision':[], 'sensitivity':[], 'specificity':[], 'f1':[], 'auc':[], 
                                'mcc':[], 'mse':[], 'misc':[]}}

    dict_metrics_test = {alphabeth : {'accuracy_train':[], 'accuracy_test':[], 'accuracy_test_ext':[], 
                                'precision':[], 'sensitivity':[], 'specificity':[], 'f1':[], 'auc':[], 
                                'mcc':[], 'mse':[], 'misc':[]}}

    for train_index, test_index in kf.split(X_main):

        X_train, X_test = X_main[train_index], X_main[test_index]
        y_train, y_test = y_main[train_index], y_main[test_index]

        #CNN------------------------------------------------------------------
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)

        n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[1], y_train.shape[1]
        input_shape = (n_features,1)

        model = create_model(input_shape, n_outputs)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

        filename = 'SavedModel/'+db+'/'+alphabeth+'_'+str(count)+'.h5'
        model.save(filename)

        (_, acc_train) = model.evaluate(X_train, y_train, verbose=0)
        (_, acc_test) = model.evaluate(X_test, y_test, verbose=0)
        dict_metrics[alphabeth]["accuracy_train"].append(acc_train)
        dict_metrics[alphabeth]["accuracy_test"].append(acc_test)

        # Predicciones
        new_predictions = model.predict(X_test, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(new_predictions, axis=1)
        y_t = np.argmax(y_test, axis=1)
        
        # MSE
        dict_metrics[alphabeth]["mse"].append(mean_squared_error(y_t, y_pred))

        # Cálculo de métricas
        cm = metrics.confusion_matrix(y_t, y_pred)
        precision, sensitivity, specificity, conf_f1, conf_misclassification = confusion_metrics(cm)

        dict_metrics[alphabeth]["precision"].append(precision)
        dict_metrics[alphabeth]["sensitivity"].append(sensitivity)
        dict_metrics[alphabeth]["specificity"].append(specificity)
        dict_metrics[alphabeth]["f1"].append(conf_f1)
        dict_metrics[alphabeth]["misc"].append(conf_misclassification)

        #ROC and AUC Score
        auc_score = roc_auc_score(y_t, y_pred)
        dict_metrics[alphabeth]["auc"].append(auc_score)

        #MCC
        mcc_score = matthews_corrcoef(y_t, y_pred)
        dict_metrics[alphabeth]["mcc"].append(mcc_score)
        
        count = count + 1
    #-------------------------------------------------------------------------------------
    # Testing

    X_testing = np.expand_dims(X_testing, axis=2)
    y_testing = to_categorical(y_testing, 2)

    for count in range (5):
        filename = 'SavedModel/'+db+'/'+alphabeth+'_'+str(count+1)+'.h5'
        new_model = keras.models.load_model(filename)
        
        # Testing Accuracy
        (_, acc_test) = new_model.evaluate(X_testing, y_testing, verbose=0)
        dict_metrics_test[alphabeth]["accuracy_train"].append(0)
        dict_metrics_test[alphabeth]["accuracy_test"].append(acc_test)

        # Predicciones
        new_predictions = new_model.predict(X_testing)
        y_pred = np.argmax(new_predictions, axis=1)
        y_t = np.argmax(y_testing, axis=1)

        # MSE
        dict_metrics_test[alphabeth]["mse"].append(mean_squared_error(y_t, y_pred))

        # Cálculo de métricas
        cm = metrics.confusion_matrix(y_t, y_pred)
        precision, sensitivity, specificity, conf_f1, conf_misclassification = confusion_metrics(cm)

        dict_metrics_test[alphabeth]["precision"].append(precision)
        dict_metrics_test[alphabeth]["sensitivity"].append(sensitivity)
        dict_metrics_test[alphabeth]["specificity"].append(specificity)
        dict_metrics_test[alphabeth]["f1"].append(conf_f1)
        dict_metrics_test[alphabeth]["misc"].append(conf_misclassification)

        #ROC and AUC Score
        auc_score = roc_auc_score(y_t, y_pred)
        dict_metrics_test[alphabeth]["auc"].append(auc_score)

        #MCC
        mcc_score = matthews_corrcoef(y_t, y_pred)
        dict_metrics_test[alphabeth]["mcc"].append(mcc_score)

    return dict_metrics, dict_metrics_test

#______________________________________________________________________________________

alphabeth = 'polarity'

dict_metrics, dict_metrics_test = model_selection(alphabeth)

print_metrics(dict_metrics[alphabeth], alphabeth, folder='Training')
print_metrics(dict_metrics_test[alphabeth], alphabeth, folder='Testing')
