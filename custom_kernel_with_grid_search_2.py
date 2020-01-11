#https://stackoverflow.com/questions/24595153/is-it-possible-to-tune-parameters-with-grid-search-for-custom-kernels-in-scikit
# list of scoring functions 
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

import sys
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import pandas as pd
from sklearn import preprocessing
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,make_scorer
from mpl_toolkits import mplot3d
import pdb
import scipy.integrate as integrate
import time
import xlsxwriter

# This file allows one to create a grid search over a custom kernel function. 
# The user need to create a class for the kernel that inherits from the 
# BaseEstimator and TransformMixin class. 
# The __init__ function should specify the parameters that need to be optimize. 
# The transform function should return the Kernel Matrix
# A pipe object need to be created for each kernel function in the main function
# as well as a dictionary of parameters. 

# Wrapper class for the custom kernel 
# that inherits from the BaseEstimator and TransformMixin class
class RBF_Kernel(BaseEstimator,TransformerMixin):
    def __init__(self, gamma=1.0):
        super(RBF_Kernel,self).__init__()
        self.gamma = gamma
        
    # Define the Kernel matrix function
    def transform(self, X):
        K_matrix = np.zeros((X.shape[0],self.X_train_.shape[0]))
        for i in range(X.shape[0]):
            x1 = X[i,:]
            for j in range(self.X_train_.shape[0]):
                x2 = self.X_train_[j,:]
                x_prime = min(abs(x1[0]-x2[0]),abs(x1[0]-x2[0]+360),abs(x1[0]-x2[0]-360))
                y_prime = min(abs(x1[1]-x2[1]),abs(x1[1]-x2[1]+360),abs(x1[1]-x2[1]-360))
                r = np.hstack((x_prime,y_prime))
                K_matrix[i,j] = np.e**-(self.gamma*(np.linalg.norm(r)**2))                # rbf
        return K_matrix
    
    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self
    
class Master_MET_ENK_kernel(BaseEstimator,TransformerMixin):
    def __init__(self,kernel ='rbf',pb ='on', gamma=1.0,beta = 1.0,coef0= 1,d = 1,gamma0 = 1.0,gamma1 =1.0,gamma2 =1.0,gamma3 =1.0,gamma4=1,gamma5=1,gamma6 =1.0,gamma7 =1.0,gamma8=1,gamma9=1):
        super(Master_MET_ENK_kernel,self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.coef0 = coef0
        self.d = d
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.gamma5 = gamma5
        self.gamma6 = gamma6
        self.gamma7 = gamma7
        self.gamma8 = gamma8
        self.gamma9 = gamma9

        
        self.pb = pb
        self.I = integrate.quad(self.integrand,-np.pi,np.pi,args=(self.gamma))[0]
        self.kernel = kernel
        
       
    def transform(self, X):
        K_matrix = np.zeros((X.shape[0],self.X_train_.shape[0]))
        for i in range(X.shape[0]):
            x1 = X[i,:]
            for j in range(self.X_train_.shape[0]):
                x2 = self.X_train_[j,:]
                if self.pb == 'on':
                    q_prime = min(abs(x1[0]-x2[0]),abs(x1[0]-x2[0]+2),abs(x1[0]-x2[0]-2))
                    r_prime = min(abs(x1[1]-x2[1]),abs(x1[1]-x2[1]+2),abs(x1[1]-x2[1]-2))
                    s_prime = min(abs(x1[2]-x2[2]),abs(x1[2]-x2[2]+2),abs(x1[2]-x2[2]-2))
                    t_prime = min(abs(x1[3]-x2[3]),abs(x1[3]-x2[3]+2),abs(x1[3]-x2[3]-2))
                    u_prime = min(abs(x1[4]-x2[4]),abs(x1[4]-x2[4]+2),abs(x1[4]-x2[4]-2))
                    v_prime = min(abs(x1[5]-x2[5]),abs(x1[5]-x2[5]+2),abs(x1[5]-x2[5]-2))
                    w_prime = min(abs(x1[6]-x2[6]),abs(x1[6]-x2[6]+2),abs(x1[6]-x2[6]-2))
                    x_prime = min(abs(x1[7]-x2[7]),abs(x1[7]-x2[7]+2),abs(x1[7]-x2[7]-2)) 
                    y_prime = min(abs(x1[8]-x2[8]),abs(x1[8]-x2[8]+2),abs(x1[8]-x2[8]-2))
                    z_prime = min(abs(x1[9]-x2[9]),abs(x1[9]-x2[9]+2),abs(x1[9]-x2[9]-2))
                    
                else:
                    q_prime = abs(x1[0]-x2[0])
                    r_prime = abs(x1[1]-x2[1])
                    s_prime = abs(x1[2]-x2[2])
                    t_prime = abs(x1[3]-x2[3])
                    u_prime = abs(x1[4]-x2[4])
                    v_prime = abs(x1[5]-x2[5])
                    w_prime = abs(x1[6]-x2[6])
                    x_prime = abs(x1[7]-x2[7])
                    y_prime = abs(x1[8]-x2[8])
                    z_prime = abs(x1[9]-x2[9])
                        
                
                r = np.hstack((x_prime,y_prime))
               
                if self.kernel == 'rbf':
                    K_matrix[i,j] = np.e**-(self.gamma*(np.linalg.norm(r)**2))
                elif self.kernel == 'joint':
                    K0 = np.e**-(self.gamma0*(np.linalg.norm(q_prime)**2))
                    K1 = np.e**-(self.gamma1*(np.linalg.norm(r_prime)**2))
                    K2 = np.e**-(self.gamma0*(np.linalg.norm(s_prime)**2))
                    K3 = np.e**-(self.gamma1*(np.linalg.norm(t_prime)**2))
                    K4 = np.e**-(self.gamma0*(np.linalg.norm(u_prime)**2))
                    K5 = np.e**-(self.gamma1*(np.linalg.norm(v_prime)**2))
                    K6 = np.e**-(self.gamma0*(np.linalg.norm(w_prime)**2))
                    K7 = np.e**-(self.gamma1*(np.linalg.norm(x_prime)**2))
                    K8 = np.e**-(self.gamma0*(np.linalg.norm(y_prime)**2))
                    K9 = np.e**-(self.gamma1*(np.linalg.norm(z_prime)**2))
                    K_matrix[i,j] = K0*K1*K2*K3*K4*K5*K6*K7*K8*K9
                elif self.kernel == 'multidimensional':
                    K0 = np.e**-(self.gamma0*(np.linalg.norm(q_prime)**2))
                    K1 = np.e**-(self.gamma1*(np.linalg.norm(r_prime)**2))
                    K2 = np.e**-(self.gamma2*(np.linalg.norm(s_prime)**2))
                    K3 = np.e**-(self.gamma3*(np.linalg.norm(t_prime)**2))
                    K4 = np.e**-(self.gamma4*(np.linalg.norm(u_prime)**2))
                    K5 = np.e**-(self.gamma5*(np.linalg.norm(v_prime)**2))
                    K6 = np.e**-(self.gamma2*(np.linalg.norm(w_prime)**2))
                    K7 = np.e**-(self.gamma3*(np.linalg.norm(x_prime)**2))
                    K8 = np.e**-(self.gamma4*(np.linalg.norm(y_prime)**2))
                    K9 = np.e**-(self.gamma5*(np.linalg.norm(z_prime)**2))
                    K_matrix[i,j] = K0*K1*K2*K3*K4*K5*K6*K7*K8*K9
                elif self.kernel == 'rq':
                     K_matrix[i,j] = (1 + (self.gamma*(np.linalg.norm(r)**2)/self.beta) )**-self.beta   
                elif self.kernel == 'poly':
                    K_matrix[i,j] = (self.gamma*np.dot(x1,x2)+ self.coef0)**self.d  
                elif self.kernel =='sigmoid':
                    K_matrix[i,j] = np.tanh((self.gamma*np.dot(x1,x2)+ self.coef0))
                elif self.kernel == 'periodic':
                     K_matrix[i,j] = np.e**(2*self.gamma*np.sin(np.deg2rad(180*np.linalg.norm(r,2)))**2)
                elif self.kernel == 'Mises':
                    K_matrix[i,j] = (1/(2*np.pi*self.I))*np.e**(self.gamma*np.cos(np.deg2rad(180*np.linalg.norm(r,2))))
        return K_matrix
    def integrand(self,x, m):
        return np.e**(-m*np.cos(x))
    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self    

class Master_Pentapeptide_kernel(BaseEstimator,TransformerMixin):
    def __init__(self,kernel ='rbf',pb ='on', gamma=1.0,beta = 1.0,coef0= 1,d = 1,gamma0 = 1.0,gamma1 =1.0,gamma2 =1.0,gamma3 =1.0,gamma4=1,gamma5=1):
        super(Master_Pentapeptide_kernel,self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.coef0 = coef0
        self.d = d
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.gamma5 = gamma5
        self.pb = pb
        self.I = integrate.quad(self.integrand,-np.pi,np.pi,args=(self.gamma))[0]
        self.kernel = kernel
        
       
    def transform(self, X):
        K_matrix = np.zeros((X.shape[0],self.X_train_.shape[0]))
        for i in range(X.shape[0]):
            x1 = X[i,:]
            for j in range(self.X_train_.shape[0]):
                x2 = self.X_train_[j,:]
                if self.pb == 'on':
                    u_prime = min(abs(x1[0]-x2[0]),abs(x1[0]-x2[0]+2),abs(x1[0]-x2[0]-2))
                    v_prime = min(abs(x1[1]-x2[1]),abs(x1[1]-x2[1]+2),abs(x1[1]-x2[1]-2))
                    w_prime = min(abs(x1[2]-x2[2]),abs(x1[2]-x2[2]+2),abs(x1[2]-x2[2]-2))
                    x_prime = min(abs(x1[3]-x2[3]),abs(x1[3]-x2[3]+2),abs(x1[3]-x2[3]-2))
                    y_prime = min(abs(x1[4]-x2[4]),abs(x1[4]-x2[4]+2),abs(x1[4]-x2[4]-2))
                    z_prime = min(abs(x1[5]-x2[5]),abs(x1[5]-x2[5]+2),abs(x1[5]-x2[5]-2))
                else:
                    u_prime = abs(x1[0]-x2[0])
                    v_prime = abs(x1[1]-x2[1])
                    w_prime = abs(x1[2]-x2[2])
                    x_prime = abs(x1[3]-x2[3])
                    y_prime = abs(x1[4]-x2[4])
                    z_prime = abs(x1[5]-x2[5])
                    
                
                r = np.hstack((x_prime,y_prime))
               
                if self.kernel == 'rbf':
                    K_matrix[i,j] = np.e**-(self.gamma*(np.linalg.norm(r)**2))
                elif self.kernel == 'joint':
                    K0 = np.e**-(self.gamma0*(np.linalg.norm(u_prime)**2))
                    K1 = np.e**-(self.gamma1*(np.linalg.norm(v_prime)**2))
                    K2 = np.e**-(self.gamma0*(np.linalg.norm(w_prime)**2))
                    K3 = np.e**-(self.gamma1*(np.linalg.norm(x_prime)**2))
                    K4 = np.e**-(self.gamma0*(np.linalg.norm(y_prime)**2))
                    K5 = np.e**-(self.gamma1*(np.linalg.norm(z_prime)**2))
                    K_matrix[i,j] = K0*K1*K2*K3*K4*K5
                elif self.kernel == 'multidimensional':
                    K0 = np.e**-(self.gamma0*(np.linalg.norm(u_prime)**2))
                    K1 = np.e**-(self.gamma1*(np.linalg.norm(v_prime)**2))
                    K2 = np.e**-(self.gamma2*(np.linalg.norm(w_prime)**2))
                    K3 = np.e**-(self.gamma3*(np.linalg.norm(x_prime)**2))
                    K4 = np.e**-(self.gamma4*(np.linalg.norm(y_prime)**2))
                    K5 = np.e**-(self.gamma5*(np.linalg.norm(z_prime)**2))
                    K_matrix[i,j] = K0*K1*K2*K3*K4*K5
                elif self.kernel == 'rq':
                     K_matrix[i,j] = (1 + (self.gamma*(np.linalg.norm(r)**2)/self.beta) )**-self.beta   
                elif self.kernel == 'poly':
                    K_matrix[i,j] = (self.gamma*np.dot(x1,x2)+ self.coef0)**self.d  
                elif self.kernel =='sigmoid':
                    K_matrix[i,j] = np.tanh((self.gamma*np.dot(x1,x2)+ self.coef0))
                elif self.kernel == 'periodic':
                     K_matrix[i,j] = np.e**(2*self.gamma*np.sin(np.deg2rad(180*np.linalg.norm(r,2)))**2)
                elif self.kernel == 'Mises':
                    K_matrix[i,j] = (1/(2*np.pi*self.I))*np.e**(self.gamma*np.cos(np.deg2rad(180*np.linalg.norm(r,2))))
        return K_matrix
    def integrand(self,x, m):
        return np.e**(-m*np.cos(x))
    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self
    
class Master_Tripeptide_kernel(BaseEstimator,TransformerMixin):
    def __init__(self,kernel ='rbf',pb ='on', gamma=1.0,beta = 1.0,coef0= 1,d = 1,gamma0 = 1.0,gamma1 =1.0,gamma2 =1.0,gamma3 =1.0):
        super(Master_Tripeptide_kernel,self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.coef0 = coef0
        self.d = d
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.pb = pb
        self.I = integrate.quad(self.integrand,-np.pi,np.pi,args=(self.gamma))[0]
        self.kernel = kernel
        
       
    def transform(self, X):
        K_matrix = np.zeros((X.shape[0],self.X_train_.shape[0]))
        for i in range(X.shape[0]):
            x1 = X[i,:]
            for j in range(self.X_train_.shape[0]):
                x2 = self.X_train_[j,:]
                if self.pb == 'on':
                    w_prime = min(abs(x1[0]-x2[0]),abs(x1[0]-x2[0]+2),abs(x1[0]-x2[0]-2))
                    x_prime = min(abs(x1[1]-x2[1]),abs(x1[1]-x2[1]+2),abs(x1[1]-x2[1]-2))
                    y_prime = min(abs(x1[2]-x2[2]),abs(x1[2]-x2[2]+2),abs(x1[2]-x2[2]-2))
                    z_prime = min(abs(x1[3]-x2[3]),abs(x1[3]-x2[3]+2),abs(x1[3]-x2[3]-2))
                else:
                    w_prime = abs(x1[0]-x2[0])
                    x_prime = abs(x1[1]-x2[1])
                    y_prime = abs(x1[2]-x2[2])
                    z_prime = abs(x1[3]-x2[3])
                    
                r = np.hstack((x_prime,y_prime))
               
                if self.kernel == 'rbf':
                    K_matrix[i,j] = np.e**-(self.gamma*(np.linalg.norm(r)**2))
                elif self.kernel == 'joint':
                    K0 = np.e**-(self.gamma0*(np.linalg.norm(w_prime)**2))
                    K1 = np.e**-(self.gamma1*(np.linalg.norm(x_prime)**2))
                    K2 = np.e**-(self.gamma0*(np.linalg.norm(y_prime)**2))
                    K3 = np.e**-(self.gamma1*(np.linalg.norm(z_prime)**2))
                    K_matrix[i,j] = K0*K1*K2*K3
                elif self.kernel == 'multidimensional':
                    K0 = np.e**-(self.gamma0*(np.linalg.norm(w_prime)**2))
                    K1 = np.e**-(self.gamma1*(np.linalg.norm(x_prime)**2))
                    K2 = np.e**-(self.gamma2*(np.linalg.norm(y_prime)**2))
                    K3 = np.e**-(self.gamma3*(np.linalg.norm(z_prime)**2))
                    K_matrix[i,j] = K0*K1*K2*K3
                elif self.kernel == 'rq':
                     K_matrix[i,j] = (1 + (self.gamma*(np.linalg.norm(r)**2)/self.beta) )**-self.beta   
                elif self.kernel == 'poly':
                    K_matrix[i,j] = (self.gamma*np.dot(x1,x2)+ self.coef0)**self.d  
                elif self.kernel =='sigmoid':
                    K_matrix[i,j] = np.tanh((self.gamma*np.dot(x1,x2)+ self.coef0))

                elif self.kernel == 'periodic':
                     K_matrix[i,j] = np.e**(2*self.gamma*np.sin(np.deg2rad(180*np.linalg.norm(r,2)))**2)
                elif self.kernel == 'Mises':
                    K_matrix[i,j] = (1/(2*np.pi*self.I))*np.e**(self.gamma*np.cos(np.deg2rad(180*np.linalg.norm(r,2))))
        return K_matrix
    def integrand(self,x, m):
        return np.e**(-m*np.cos(x))
    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self

class Master_Dipeptide_kernel(BaseEstimator,TransformerMixin):
    def __init__(self,kernel ='rbf',pb ='on', gamma=1.0,beta = 1.0,coef0= 1,d = 1,gamma0 = 1.0,gamma1 =1.0):
        super(Master_Dipeptide_kernel,self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.coef0 = coef0
        self.d = d
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.pb = pb
        self.kernel = kernel
        self.I = integrate.quad(self.integrand,-np.pi,np.pi,args=(self.gamma))[0]
       
    def transform(self, X):
        K_matrix = np.zeros((X.shape[0],self.X_train_.shape[0]))
        for i in range(X.shape[0]):
            x1 = X[i,:]
            for j in range(self.X_train_.shape[0]):
                x2 = self.X_train_[j,:]
                if self.pb == 'on':
                    x_prime = min(abs(x1[0]-x2[0]),abs(x1[0]-x2[0]+2),abs(x1[0]-x2[0]-2))
                    y_prime = min(abs(x1[1]-x2[1]),abs(x1[1]-x2[1]+2),abs(x1[1]-x2[1]-2))
                else: 
                     x_prime = abs(x1[0]-x2[0])
                     y_prime = abs(x1[1]-x2[1])
                r = np.hstack((x_prime,y_prime))
               
                if self.kernel == 'rbf':
                    K_matrix[i,j] = np.e**-(self.gamma*(np.linalg.norm(r)**2))
                elif self.kernel == 'joint' or self.kernel == 'multidimesional':
                    K0 = np.e**-(self.gamma0*(np.linalg.norm(x_prime)**2))
                    K1 = np.e**-(self.gamma1*(np.linalg.norm(y_prime)**2))
                    K_matrix[i,j] = K0*K1
                elif self.kernel == 'rq':
                     K_matrix[i,j] = (1 + (self.gamma*(np.linalg.norm(r)**2)/self.beta) )**-self.beta   
                elif self.kernel == 'poly':
                    K_matrix[i,j] = (self.gamma*np.dot(x1,x2)+ self.coef0)**self.d  
                elif self.kernel =='sigmoid':
                    K_matrix[i,j] = np.tanh((self.gamma*np.dot(x1,x2)+ self.coef0))
                elif self.kernel == 'periodic':
                     K_matrix[i,j] = np.e**(2*self.gamma*np.sin(np.deg2rad(180*np.linalg.norm(r,2)))**2)
                elif self.kernel == 'Mises':
                    K_matrix[i,j] = (1/(2*np.pi*self.I))*np.e**(self.gamma*np.cos(np.deg2rad(180*np.linalg.norm(r,2))))
        return K_matrix
    def integrand(self,x, m):
        return np.e**(-m*np.cos(x))
    
    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self
# This functions controls the file path for the full training data sets and 
# test sets
def csv_open(data_set):
    if data_set == 'dipeptide':
        training = pd.read_csv('Dipeptide_data/training_20000_random.csv')
        testing = pd.read_csv('Dipeptide_data/Test_set.csv') 
        dimension = 2
    elif data_set == 'tripeptide':
        training = pd.read_csv('Tripeptide_data/tripeptide_training_full_random.csv',header = None)
        testing = pd.read_csv('Tripeptide_data/tripeptide_test_set.csv',header = None) 
        dimension = 4
    elif data_set == 'pentapeptide':
        training = pd.read_csv('Pentapeptide_data/pentapeptide_training_full_random.csv',header = None)
        testing = pd.read_csv('Pentapeptide_data/pentapeptide_test_set.csv',header = None) 
        training.iloc[:,:-1] =  training.iloc[:,:-1]*180/np.pi
        testing.iloc[:,:-1] =  testing.iloc[:,:-1]*180/np.pi
        dimension = 6
    elif data_set == 'MET-ENK':
        training = pd.read_csv('MET-ENK_data/MET-ENK_training_300_random.csv',header = None)
        testing = pd.read_csv('MET-ENK_data/MET-ENK_test_set.csv',header = None)
        training.iloc[:,:-1] =  training.iloc[:,:-1]*180/np.pi
        testing.iloc[:,:-1] =  testing.iloc[:,:-1]*180/np.pi
        dimension = 10
    return training,testing,dimension

# creates a 3D grid for the result data fram
def create3d(result,cv_value,N,data_set):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(result['param_per_func__gamma'], result['param_svm__C'], result['param_svm__epsilon'], c=-result['mean_test_score'], cmap='viridis')
    cb = fig.colorbar(p) 
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    ax.set_zlabel('epsilon')
    ax.set_title(' MSE surface CV = '+str(cv_value)+'Using N ='+str(N))
#    plt.savefig('cross-validation/'+data_set+'/N_'+str(N)+'/MSE_Surface_cv_' + str(cv_value)+'.png')

# The normalization function below manually normalized the training data 
# and the dihedral columns of the testing set
def normalization(training_data,testing_data):

    # Compute the mean and stdev of each column
    training_mean = training_data.mean(axis=0)
    training_stdev = training_data.std(axis=0)
    
    
    training_data = (training_data - training_mean)  # subtract mean from each column
    training_data.iloc[:,:-1] = training_data.iloc[:,:-1]/180 # divide the angle columns by the standard deviation
    training_data.iloc[:,-1] = training_data.iloc[:,-1]/training_stdev.iloc[-1] # divide the last column by its stdev
    testing_data.iloc[:,:-1] = (testing_data.iloc[:,:-1] - training_mean.iloc[:-1])/180 # subtract the training mean and divide by the angle stdev
    
    # convert dataframes to numpy arrays
    training_mean = training_mean.values
    training_data = training_data.values
    testing_data = testing_data.values
    training_stdev = training_stdev.values

    return training_data,testing_data,training_stdev,training_mean # all numpy arrays

def C_MSE_plot(result,param,data_set,pb,N,cv_value,epsilon):
    plt.figure()
    for C in param['svm__C']:
        q = 'param_svm__C == ' + str(C) + ' and param_svm__epsilon == '+str(epsilon) # define query`
        plt.plot(result.query(q)['param_dp_func__gamma'],-result.query(q)['mean_test_score'],label = 'C = ' + str(C))
    plt.xlabel('gamma')
    plt.ylabel('Mean MSE')
    plt.title('MSE Dependence on C, CV = ' + str(cv_value)+ ' Using N = '+ str(N))
    plt.legend()
    if pb == 'on':
        plt.savefig('cross-validation/'+data_set+'/PBC/N_'+str(N)+'/MSE_Surface_cv_' + str(cv_value)+'_C.png')
    else:
        plt.savefig('cross-validation/'+data_set+'/NO_PBC/N_'+str(N)+'/MSE_Surface_cv_' + str(cv_value)+'_C.png')

def epsilon_MSE_plot(result,param,data_set,pb,N,cv_value,C):
    plt.figure()
    for epsilon in param['svm__epsilon']:
        q = 'param_svm__epsilon == ' + str(epsilon) + ' and param_svm__C == '+str(C)# define query`
        plt.plot(result.query(q)['param_dp_func__gamma'],-result.query(q)['mean_test_score'],label = 'epsilon = ' + str(epsilon))
    plt.xlabel('gamma')
    plt.ylabel('Mean MSE')
    plt.title('MSE Dependence on Epsilon, CV = ' + str(cv_value)+' Using N = '+ str(N))
    plt.legend()
    if pb == 'on':
        plt.savefig('cross-validation/' +data_set+'/PBC/N_'+str(N)+'/MSE_Surface_cv_' + str(cv_value)+'_epsilon.png')
    else:
        plt.savefig('cross-validation/' +data_set+'/NO_PBC/N_'+str(N)+'/MSE_Surface_cv_' + str(cv_value)+'_epsilon.png')
            
def append_results_to_excel(best_results_list,model,full_path,param_dict,kernel):

    param_list=[]
    for param in model.best_params_.keys():
        name = param[param.index('__')+2:]
        param_list.append(name)
        
    column_names = ['dataset','Kernel','N','CV'] + param_list +['CV MSE','CV time (min)']
    best_result_frame = pd.DataFrame(data = best_results_list,columns=column_names) # create dataframe
    best_result_frame.drop(columns = ['kernel'],inplace = True)
    print(best_result_frame)
    
    pf = pd.DataFrame(list(param_dict.values()),# create dataframe storing parameter dictionary
                                   index=param_list)
    pf.drop(labels = ['kernel'],axis = 0,inplace = True)
    pf.fillna('',inplace=True)
    
    writer = pd.ExcelWriter(full_path, engine = 'xlsxwriter') # create writer object
    best_result_frame.to_excel(excel_writer = writer,sheet_name = kernel)
    pf.to_excel(excel_writer = writer,sheet_name = 'grid')
    writer.save()
    writer.close()
    
  
def main():

    print('python: {}'.format(sys.version))
    print('numpy: {}'.format(np.__version__))
    print('sklearn: {}'.format(sklearn.__version__))
    np.random.seed(0)

    data_set = 'dipeptide'
    training,testing,dim = csv_open(data_set)
    training, testing, training_stdev,training_mean = normalization(training,testing)
    pb = 'off'
    kernel = 'rbf'
    plt.close('all')
    excel_path = 'C:/cygwin64/home/jaipe/Thesis/cross-validation/CV_results/'
    full_path = excel_path + data_set+'_'+kernel+'_PBC'+pb+'.xlsx'
    
    best_results_list = [] # this list stores the best result for each trial

    datapoints = [100]
    for N in datapoints:
        X_train = training[0:N,0:dim]
        y_train = training[0:N,dim]
    
        
    
        # Inorder to perform a customized grid search, A kernel class like the ones
        # above need to be created. A pipeline object need to be created for each
        # kernel as well as a dictionary of parameters as shown below.
        rbf_pipe = Pipeline([
            ('rbf_func',RBF_Kernel()), # name of the kernel being tested, class name of kernel
            ('svm', SVR()),            # model to be passed in  
        ])
        
    
        dipeptide_pipe = Pipeline([
            ('dp_func',Master_Dipeptide_kernel(kernel=kernel,pb=pb)),
            ('svm', SVR(C= 10, epsilon = 0.01)),    
        ])
    
        tripeptide_pipe = Pipeline([
            ('tri_func',Master_Tripeptide_kernel(kernel=kernel)),
            ('svm', SVR(C= 10, epsilon = 0.01)),    
        ])
    
        pentapeptide_pipe = Pipeline([
            ('penta_func',Master_Pentapeptide_kernel(kernel=kernel, pb =pb)),
            ('svm', SVR(C= 10, epsilon = 0.01)),    
        ])
        met_enk_pipe = Pipeline([
            ('met_func',Master_MET_ENK_kernel(kernel=kernel, pb =pb)),
            ('svm', SVR(C= 10, epsilon = 0.01)),    
        ])
        
        
        # Set the parameter 'gamma' of our custom kernel by
        # using the 'estimator__param' syntax.
        rbf_params = dict([
            ('dp_func__gamma', (0.001,0.01,0.1,0.2,0.6,1,2,4,6,8,10)), # name__parameter
            ('svm__C',(0.1,1,2,4,6,8,10)),
            ('svm__epsilon',(0.001,0.005,0.01,1,2)),
            ('svm__kernel', ['precomputed']), # notify the model that the kernel matrix is precomputed
        ])
        
    
        dipeptide_params = dict([
            ('dp_func__gamma', (0.1,1,2,4,6,8,10)), # name__parameter
#            ('svm__epsilon',(0.001,0.005,0.01,1,2,4,6,8,10)),
            ('svm__C',(0.001,0.01,0.1,0.2,0.6,1,1.2,1.4,1.6,2,4,6,8,10)),
            ('svm__kernel', ['precomputed']), # notify the model that the kernel matrix is precomputed
        ])
    
        tripeptide_params = dict([
            ('tri_func__gamma0', (0,5,10,15,20,25)), # name__parameter
            ('tri_func__gamma1', (0,5,10,15,20,25)),
            ('tri_func__gamma2', (0,5,10,15,20,25)),
            ('tri_func__gamma3', (0,5,10,15,20,25)),
            ('svm__epsilon',(.1,0.3,0.5)),
            ('svm__kernel', ['precomputed']), # notify the model that the kernel matrix is precomputed
        ])
    
        pentapeptide_params = dict([
            ('penta_func__gamma', (0.001,0.01,0.1,0.2,0.6,1,1.2,1.4,1.6,2,4,6,8,10)), # name__parameter
            ('svm__epsilon', (0.01,1)),
            ('svm__C',(0.1,1,2,4,6,8,10)),
            ('svm__kernel', ['precomputed']), # notify the model that the kernel matrix is precomputed
        ])
        met_enk_params = dict([
            ('met_func__gamma0', (0.2,0.6,1,1.4,1.8,2.2,2.6,3,3.4,3.8,4.2,4.6,5,5.4,5.8,6.2,6.6,7)), # name__parameter
            ('met_func__gamma1', (0.2,0.6,1,1.4,1.8,2.2,2.6,3,3.4,3.8,4.2,4.6,5,5.4,5.8,6.2,6.6,7)),
            ('svm__epsilon',(0.001,0.01,0.1,0.3,1)),
            ('svm__C',(1,2,4,6,8,10)),
            ('svm__kernel', ['precomputed']), # notify the model that the kernel matrix is precomputed
        ])
        
        
        
        # collect all references to the pipes and parameters for each kernel that need to be tested
        pipes = [dipeptide_pipe]# poly_pipe, sig_pipe,periodic_pipe,rq_pipe] 
        cv_params = [dipeptide_params]#poly_params,sig_params,periodic_params,rq_params]
        


        # Do grid search to get the best parameter value of 'gamma'.
        cv_list = [2,3]
        for cv_value in cv_list: # For loop to determine the number of cross validations
            
            for pipe,param in zip(pipes,cv_params):
                print('---------------------------')
                print('For N = ', N, ' cv =', cv_value)
                start_time = time.time()
                model = GridSearchCV(pipe, param, cv=cv_value, verbose=1, n_jobs=-1,scoring= 'neg_mean_squared_error')
                model.fit(X_train, y_train)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore") # supress warnings 
                    result = pd.DataFrame(model.cv_results_)
                pd.set_option('display.max_columns', None) # sets max display column to none 
                #print(result[['param_rbf_func__gamma','param_svm__C','mean_test_score','rank_test_score']])
                #print(result.head())

                
                
#                create3d(result,cv_value,N,data_set) # creates a 3D plot of gamma,C,epsilon with MSE as the color dimension
#                C_MSE_plot(result,param,data_set,pb,N,cv_value,1)        # creates a 2D plot of the MSE as a function of gamma across varying C
#                epsilon_MSE_plot(result,param,data_set,pb,N,cv_value,10) # creates a 2D plot of the MSE as a function of gamma across varying epsilon
                elapsed_time = round((time.time() - start_time)/60,2)
                temp = [data_set,kernel,N,cv_value] + list(model.best_params_.values())+[round(-model.best_score_,2),elapsed_time] # add data point an cv to list
                print(temp)
                best_results_list.append(temp)
    
    # compute the number of candidates multiplied by the number of data points
    # multiplied by the number of cv values used
    # to to help seperate results based on the grid used
    candidates = result.shape[0] 
    full_path = excel_path + data_set+'_'+kernel+'_PBC'+pb+'_'+ str(len(datapoints)*len(cv_list)*candidates)+'.xlsx'
    append_results_to_excel(best_results_list,model,full_path,dipeptide_params,kernel)
    
  

if __name__ == '__main__':
    main()