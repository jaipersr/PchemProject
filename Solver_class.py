# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:06:09 2019

@author: jaipe
"""
import numpy as np
from scipy import optimize
import time
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVR
import multiprocessing as mp
from obj_class import objective_class
import pdb
import scipy.integrate as integrate



class Solver():
    def __init__(self,N,orig_trainining,orig_testing,X_train,Y_train,X_test,Y_test,kernel,training_stdev = [1],training_mean = [0],C=10,epsilon=1,gamma=1,gamma_vector=[],sigma =1,p =1,pb = 'on',Warning_message = 'on'):
        self.Warning_message = Warning_message
        self.N = N
        self.orig_training = orig_training
        self.orig_testing = orig_testing
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.kernel = kernel
        self.training_stdev = training_stdev
        self.training_mean = training_mean
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.gamma_vector = gamma_vector
        self.sigma = sigma                      # parameter for hubert loss objective function
        self.p = p                              # parameter for polynomial and piecewise objective functions 
        self.pb = pb
        self.I = integrate.quad(self.integrand,-np.pi,np.pi,args = (self.gamma))[0]
        self.I2 = integrate.quad(self.integrand,0,2*np.pi,args = (self.gamma))[0]
        self.K_matrix = self.Kernel_matrix()    # compute the Kernel matrix
        
    
    # creates the Kernel matrix by making calls to the Kernel function
    def Kernel_matrix(self):
        K = np.zeros((self.X_train.shape[0],self.X_train.shape[0]))
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                K[i,j] = self.Kernel_function(self.X_train[i,:],self.X_train[j,:])
        
        if self.Warning_message == 'on':
            self.is_psd(K)
            print('K_matrix size',K.shape)
            print('K sum: ', round(K.sum(),2))
        return K 
    
    def Kernel_matrix_pred(self,X_test):
        K = np.zeros((X_test.shape[0],self.X_train.shape[0]))
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                K[i,j] = self.Kernel_function(X_test[i,:],self.X_train[j,:])
        print('K_matrix_pred size',K.shape)
        return_val = self.is_psd(K)
        return K
    
    def integrand(self,x, m):
        return np.e**(-m*np.cos(x))
    
    # defines the kernel function
    def Kernel_function(self,x1,x2):
        
        r_dist = np.zeros_like(x1) 
        for i in range(x1.shape[0]):
            if i == 4:
                break
            if self.pb == 'on':
                 nearest_image = min(abs(x1[i]-x2[i]),abs(x1[i]-x2[i]+2),abs(x1[i]-x2[i]-2))
            else:
                nearest_image = abs(x1[i]-x2[i]) 
            r_dist[i] = nearest_image 
            
        if self.kernel == 'rbf':
            K_ij = np.e**-(self.gamma*(np.linalg.norm(r_dist)**2))
        elif self.kernel == 'multidimensional':
            K_ij = 1        
            for i in range(r_dist.shape[0]):
                K_ij *= np.e**-(self.gamma_vector[i]*(np.linalg.norm(r_dist[i])**2))
        elif kernel == 'periodic':
            K_ij = np.e**-(2*self.gamma*np.sin(np.deg2rad((180*np.linalg.norm(r_dist,1)/2)))**2)
        elif kernel == 'Mises':
            A = (1/(2*np.pi*self.I))
            K_ij = A*np.e**(self.gamma*np.cos(np.deg2rad(180*np.linalg.norm(r_dist,2))))
        elif kernel == 'Mises2':
                        
            A = (1/(2*np.pi*self.I2))
            
            if r_dist[0] == 0 and r_dist[1] == [0]:
                pass
            else:
                r_dist = r_dist/((r_dist[0]**2 + r_dist[1]**2)**0.5)
                
            if r_dist[0] == 0:
                if r_dist[1] > 0:
                    theta = np.pi/2
                else:
                    theta = 3*np.pi/2
            elif r_dist[0] < 0:
                theta = np.arctan(r_dist[1]/r_dist[0]) + np.pi
            elif r_dist[1] < 0:
                theta = np.arctan(r_dist[1]/r_dist[0]) + 2*np.pi
            else:
                theta = np.arctan(r_dist[1]/r_dist[0])
                
            K_ij = A*np.e**(self.gamma*np.cos(theta))
        
        elif kernel == 'joint':
            K_ij = 1 
            for i in range(r_dist.shape[0]):
                if i%2 == 0:
                    K_ij *= np.e**-(self.gamma_vector[0]*(np.linalg.norm(r_dist[i])**2))
                else:
                    K_ij *= np.e**-(self.gamma_vector[1]*(np.linalg.norm(r_dist[i])**2))
        return K_ij
    
    def is_psd(self,K):
        if np.all(np.linalg.eigvals(K) > 0):
            #print('Kernel matrix is pd')
            return 1
        elif np.all(np.linalg.eigvals(K) >= 0):
            #print('Kernel matrix is psd')
            return 0 
        else:
            #print('Kernel matrix is not psd')
            return -1
        
    def is_psd2(self):
        if np.all(np.linalg.eigvals(self.K_matrix) > 0):
            #print('Kernel matrix is pd')
            return 'PD'
        elif np.all(np.linalg.eigvals(self.K_matrix) >= 0):
            #print('Kernel matrix is psd')
            return 'PSD' 
        else:
            #print('Kernel matrix is not psd')
            return 'Not PD'

    
    def sum_constraint(self,alpha_total):
        alpha,alpha_s = np.split(alpha_total, 2)
        return alpha.sum() - alpha_s.sum()
    
    # This function computes the intercept outlined by SMOLA in section 1.4
    def compute_b(self):
        w = np.zeros((1,self.X_train.shape[1]))
        for i in range(self.X_train.shape[0]):
            w += (self.alpha[i] - self.alpha_s[i])*self.X_train[i] # equation 11 from Smola to compute w
        
        # create an empty sequence to store values, create an initial min and max value
        lower_bound = np.array([])
        upper_bound = np.array([])
        
        # Follow equation 16 from SMOLA 
        for i in range(self.X_train.shape[0]):
            value = -self.epsilon + self.Y_train[i] - np.dot(w,self.X_train[i])
            if self.alpha[i] < self.C or self.alpha_s[i] > 0: # compute max value of lower bound
                lower_bound = np.append(lower_bound,value)    
            if self.alpha[i] > 0 or self.alpha_s[i] < self.C : # compute min value of upper bound
                upper_bound = np.append(upper_bound,value)    
        self.b = (max(lower_bound) + min(upper_bound)) /2  # compute b by taking the average of the min and max
        return
    
    def compute_weight(self):
        self.weights = np.dot(self.dual_coef_.T,self.X_train)
        return
    
    def compute_KRR(self,L):
        self.KRR_alpha = np.dot(np.linalg.inv(self.K_matrix - L*np.identity(self.K_matrix.shape[0])),self.Y_train)
        return 
    
    def decision_KRR(self,new_x):
        prediction = 0
        for i in range(self.X_train.shape[0]):
            prediction += self.KRR_alpha[i]*np.e**-(self.gamma*(np.linalg.norm(self.X_train[i] -new_x)**2))
        prediction = prediction*self.training_stdev[-1] + self.training_mean[-1] # invert the transform to get the prediction data un scaled
        
        return prediction
            
        
    
        # decision function for computing predictions
    def decision_function(self,new_x):
        prediction = 0
        for i in range(self.X_train.shape[0]):
            prediction += self.dual_coef_[i] * self.Kernel_function(self.X_train[i],new_x)
        #prediction += self.b
        prediction = prediction*self.training_stdev[-1] + self.training_mean[-1] # invert the transform to get the prediction data un scaled
        return prediction
    
    # This function counts the number of support vectors from a result
    def count_support_vector(self):
        soln = np.hstack((self.alpha,self.alpha_s))
        num = soln.shape[0]
        sv_index = []
        non_sv_index = []
        for i in range(soln.shape[0]):
            if soln[i,0] == 0 and soln[i,1] == 0: # check the number of nonzero alpha pairs
                num -= 1
                non_sv_index.append(i)
            else:
                sv_index.append(i)
        return num, sv_index,non_sv_index
    
    # Since most of the alpha and alpha_s in result vector are small 
    # this sets them to zero
    def set_low_alpa_to_zero(self):
        for i in range(self.result.x.shape[0]):
            if self.result.x[i] < 0.01:
                self.result.x[i] = 0
    
    # This function computes the energy at each grid point for 
    # Alanine Dipeptide exclusively
    def create_grid(self,MSE,resolution,problem,scatter ='off'):
        
        if resolution == 'high':
            x = y = np.arange(-180,180,2)
            X,Y = np.meshgrid(x,y)
            Z = np.zeros((180,180))
        if resolution == 'low':
            x = y = np.arange(-180,180,4)
            X,Y = np.meshgrid(x,y)
            Z = np.zeros((90,90))
            
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # normalize input 
                X_norm = (X[i,j] - self.training_mean[0])/180 #self.training_stdev[0] 
                Y_norm = (Y[i,j] - self.training_mean[1])/180 #self.training_stdev[1]
                scaled_grid_point = np.reshape([(X_norm,Y_norm)],(2,1))
                if problem == 'original':
                    prediction = self.decision_function(scaled_grid_point)
                elif problem == 'KRR':
                    prediction = self.decision_KRR(scaled_grid_point)
                Z[i,j] = prediction
    
        fig, ax = plt.subplots()
        p = ax.pcolor(Y, X, Z, vmin=Z.min(), vmax=Z.max()) # pass psi as the xlabel and phi as the y label
        plt.contour(Y, X, Z)
        cb = fig.colorbar(p)
        
        if scatter == 'all':
            plt.scatter(self.orig_training.iloc[0:self.N,1],self.orig_training.iloc[0:self.N,0],marker = 'o',color='black')
        elif scatter == 'sv':
            scatter_lst = []
            for i in self.sv_index:
                scatter_lst.append([self.orig_training.iloc[i,0],self.orig_training.iloc[i,1]])
            scatter_lst = np.array(scatter_lst)
            plt.scatter(scatter_lst[:,1],scatter_lst[:,0],color = 'red')
        # this elif will display all points and label sv red and non sv black
        elif scatter == 'lp': # labeled points
            sv_lst = []
            nonsv_lst = []
            for i in range(self.N):
                if i in self.sv_index:
                    sv_lst.append([self.orig_training.iloc[i,0],self.orig_training.iloc[i,1]])
                else:
                    nonsv_lst.append([self.orig_training.iloc[i,0],self.orig_training.iloc[i,1]])
            sv_lst = np.array(sv_lst)
            nonsv_lst = np.array(nonsv_lst)
            plt.scatter(sv_lst[:,1],sv_lst[:,0],color = 'red')
            
            if nonsv_lst.shape[0] != 0:
                plt.scatter(nonsv_lst[:,1],nonsv_lst[:,0],color = 'black')
            
        
        plt.xlabel('Psi')
        plt.ylabel('Phi')
        plt.xlim(xmin=-180,xmax=180)
        plt.ylim(ymin=-180,ymax=180)
        cb.set_label('Helmholtz Energy', rotation=270)
        plt.title('FES of Alanine Dipeptide Using ' + self.kernel+' MSE = ' +str(round(MSE,3)) + ', gamma =' +str(round(self.gamma,2)))
    
    def yhat(self,Y_pred):
        x = np.linspace(np.amin(self.orig_training.iloc[0:self.N,-1]),np.amax(self.orig_training.iloc[0:self.N,-1]),100)
        y = x
        plt.figure()
        plt.plot(x,y)
        plt.scatter(Y_pred,self.Y_test,marker ='o')
        plt.xlabel('Y_pred')
        plt.ylabel('Y_test')
        plt.title('Yhat Plot N = '+str(self.N))
        
    def create_histogram(self):
        plt.figure()
        plt.hist(self.dual_coef_,bins='auto')
        plt.xlabel('Dual Coefficents')
        plt.ylabel('Frequency')
        plt.title('Distributions of alpha differences')
    
    def create_histogram2(self,x,dim):
        plt.figure()
        plt.hist(x,bins='auto')
        if dim%2 == 0:
            plt.xlabel('Phi_' + str(int(dim/2)))
        else:
            plt.xlabel('Psi_' + str(int(dim/2)))
        plt.ylabel('Frequency')
        plt.title('Distributions of angle values for dim = '+str(dim))

    def minimize(self,data_set,obj = 'epsilon_sensitive',FES='off',Yhat='off',dim_dist ='off',alpha_dist  = 'off'):
        # Put alphas into a single vector
        alpha_total = self.C*np.ones((2*self.X_train.shape[0],1)) # alpha_total = [alpha,alpha_s]
        cons = [{"type": "ineq", "fun": lambda alpha_total: alpha_total}, # alpha >= 0 , alpha_s >= 0
                {"type": "ineq", "fun": lambda alpha_total: self.C - alpha_total}, # C - alpha >=0, C - alpha_s >=0
                {"type": "eq",   "fun": self.sum_constraint}] 
    

        print('objective function = ', obj)
        start_time = time.clock()
        instance = objective_class(self.K_matrix,self.Y_train,C = self.C,epsilon = self.epsilon,sigma = self.sigma, p =self.p)
        obj_func = instance.get_function(obj = obj)
        grad = instance.get_grad(obj = obj)
        
        # Minimize the objective function
        # The result vector is the soln of alpha and alpha_s 
        self.result = optimize.minimize(obj_func,alpha_total,method = 'SLSQP',jac=grad,constraints = cons,tol = 0.001,options={'maxiter':1000})    
        end_time = time.clock() 
        total_time = round((end_time - start_time)/60,2)
    
        #-----------------------------------------------------------------------------#
        print('Total Time to Train model: ',total_time,' min')
        print('Status of optimization',self.result.success) 
        print('Number of iterations',self.result.nit)
        print('Message',self.result.message)
        #-----------------------------------------------------------------------------#
        
        

        self.set_low_alpa_to_zero()
        # Most alphas are close to zero or lie on the boundary C 
        # there can never be a set of alpha and alpha_s which are simultaneously non zero
        alpha,alpha_s = np.split(self.result.x, 2)
        self.alpha = np.resize(alpha,(alpha.shape[0],1))
        self.alpha_s = np.resize(alpha_s,(alpha_s.shape[0],1)) 
        self.dual_coef_ = alpha - alpha_s
        Support_Vectors = np.array(self.X_train)
        
        self.num_sv,self.sv_index,self.non_sv_index = self.count_support_vector()
        print('Number of support vectors',self.num_sv)
        print('result.x',self.result.x.shape)
        print('dual_coef_', self.dual_coef_.shape)
        print('Support_Vectors', Support_Vectors.shape)
                
        
        self.compute_b()
        self.compute_weight()
        print('weights',self.weights)
        print('intercept b',round(self.b,2))
        
        Y_pred = np.zeros((self.X_test.shape[0],1))
        
        for i in range(self.X_test.shape[0]):
            Y_pred[i] = self.decision_function(self.X_test[i,:]) # compute the prediction
        MSE = mean_squared_error(self.Y_test, Y_pred) # compute the MSE
        MSE = round(MSE,3)
        L2 = round(MSE**0.5,3)
        print('MSE:', MSE)
        print('L2:',L2)
        
        
        
        
#        L = 2
#        self.compute_KRR(L)
#        Y_pred_KRR = np.zeros((self.X_test.shape[0],1))
#        for i in range(self.X_test.shape[0]):
#            Y_pred_KRR[i] = self.decision_KRR(self.X_test[i,:])
#        MSE2 = mean_squared_error(self.Y_test, Y_pred_KRR)
#        print('KRR MSE: ', round(MSE2,3))
#        print('KRR L2: ',round(MSE2**0.5,3))
        
        
        if Yhat == 'on':
            self.yhat(Y_pred)
            #self.create_histogram()
        if dim_dist == 'on':
            for i in range(self.X_train.shape[1]):
                self.create_histogram2(self.orig_training.iloc[0:self.N,i],i)
            
        if FES == 'on' and data_set == 'dipeptide':
            self.create_grid(MSE,resolution ='low',problem = 'original',scatter = 'lp')

        if alpha_dist  == 'on':
            self.create_histogram()
#            self.create_grid(MSE,'low','KRR')
            
        return MSE,L2
            
        
        




# This functions controls the file path for the full training data sets and 
# test sets
def csv_open(data_set):
    if data_set == 'dipeptide':
        training = pd.read_csv('Dipeptide_data/training_20000_random.csv',header = None)
        testing = pd.read_csv('Dipeptide_data/Test_set.csv',header = None) 
        dimension = 2
    elif data_set == 'tripeptide':
        training = pd.read_csv('Tripeptide_data/tripeptide_training_full_random.csv',header = None)
        testing = pd.read_csv('Tripeptide_data/tripeptide_test_set.csv',header = None) 
        dimension = 4
    elif data_set == 'pentapeptide':
        training = pd.read_csv('Pentapeptide_data/pentapeptide_training_full_random.csv',header = None)
        testing = pd.read_csv('Pentapeptide_data/pentapeptide_test_set.csv',header = None) 
        training.iloc[:,0:-1] = training.iloc[:,0:-1]*180/np.pi
        testing.iloc[:,0:-1] = testing.iloc[:,0:-1]*180/np.pi
        dimension = 6
    elif data_set == 'MET-ENK':
        training = pd.read_csv('MET-ENK_data/MET-ENK_training_300_random.csv',header = None)
        testing = pd.read_csv('MET-ENK_data/MET-ENK_test_set.csv',header = None)
        training.iloc[:,0:-1] = training.iloc[:,0:-1]*180/np.pi
        testing.iloc[:,0:-1] = testing.iloc[:,0:-1]*180/np.pi
        dimension = 10
    return training,testing,dimension

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

# The normalization function below utilizes sklearn to normalize
# the training data and the dihedral columns of the testing set
def normalization2(training_data,testing_data):

    scaler = preprocessing.StandardScaler() # creates object that with hold mean, stdev to scale the data
    training_data = scaler.fit_transform(training_data ) # scales the training data
    training_mean = scaler.mean_ # generates the mean for each feature
    training_stdev = scaler.var_**0.5 # generate the stdev for each feature
    testing_data = scaler.transform(testing_data) # uses the mean and stdev of the training data to scale the testing data
    testing_data[:,-1]  =  testing_data[:,-1]*training_stdev[-1] + training_mean[-1] # invert the transform to get the test data un scaled
    return training_data,testing_data,training_stdev,training_mean # all nu

# Creates a 3D grid that will show regions K is PD for the RBF kernel
def create3d_rbf(N,orig_training,orig_testing,X_train,Y_train,X_test,Y_test,kernel,pb,training_stdev,training_mean,g_lst,epsilon_lst,C_lst):
    grid_matrix = np.array([[0,0,0,0]])
    for g in g_lst:
        for epsilon in epsilon_lst:
            for C in C_lst:        
                soln = Solver(N,orig_training,orig_testing,X_train,Y_train,X_test,Y_test,kernel,training_stdev,training_mean,\
                      C=C,epsilon=epsilon,gamma=g,gamma_vector = [1,1], sigma = 1,p =1,pb= pb,Warning_message = 'off')
        
                if soln.is_psd(soln.K_matrix) == 1: 
                    grid_matrix = np.vstack((grid_matrix,np.array([g,C,epsilon,1]))) 
                elif soln.is_psd(soln.K_matrix) == 0:
                    grid_matrix = np.vstack((grid_matrix,np.array([g,C,epsilon,0])))
                else:
                    grid_matrix = np.vstack((grid_matrix,np.array([g,C,epsilon,-1])))
    grid_matrix = grid_matrix[1:,:]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(grid_matrix[:,0],grid_matrix[:,1] ,grid_matrix[:,2] , c=grid_matrix[:,3], cmap='viridis')
    cb = fig.colorbar(p) 
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    ax.set_zlabel('epsilon')
    ax.set_title(' PD surface Using N ='+str(N))

# Creates a 3D grid that will show regions K is PD for the RBF kernel
def create3d_joint(N,orig_training,orig_testing,X_train,Y_train,X_test,Y_test,kernel,pb,training_stdev,training_mean,g0_lst,g1_lst,epsilon_lst,C_param):
    C= C_param
    grid_matrix = np.array([[0,0,0,0]])
    for g0 in g0_lst:
        for g1 in g1_lst:
            for epsilon in epsilon_lst:
                soln = Solver(N,orig_training,orig_testing,X_train,Y_train,X_test,Y_test,kernel,training_stdev,training_mean,\
                      C=C,epsilon=epsilon,gamma=1,gamma_vector = [g0,g1], sigma = 1,p =1,pb= pb,Warning_message = 'off')
        
                if soln.is_psd(soln.K_matrix) == 1: 
                    grid_matrix = np.vstack((grid_matrix,np.array([g0,g1,epsilon,1]))) 
                elif soln.is_psd(soln.K_matrix) == 0:
                    grid_matrix = np.vstack((grid_matrix,np.array([g0,g1,epsilon,0])))
                else:
                    grid_matrix = np.vstack((grid_matrix,np.array([g0,g1,epsilon,-1])))
    grid_matrix = grid_matrix[1:,:]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(grid_matrix[:,0],grid_matrix[:,1] ,grid_matrix[:,2] , c=grid_matrix[:,3], cmap='viridis')
    cb = fig.colorbar(p) 
    ax.set_xlabel('g0')
    ax.set_ylabel('g1')
    ax.set_zlabel('epsilon')
    ax.set_title(' PD surface Using N ='+str(N)+' PB = ' + pb)

def append_solver_results_to_excel(results_list,full_path,kernel):
    
    column_names = ['N','PBC','C','epsilon','g','Solver MSE (kcal/mol)^2','Solver L2 (kcal/mol)','PD Nature']
    result_frame = pd.DataFrame(data = results_list,columns=column_names) # create dataframe
    print(result_frame)
    
    writer = pd.ExcelWriter(full_path, engine = 'xlsxwriter') # create writer object
    result_frame.to_excel(excel_writer = writer,sheet_name = kernel)

    writer.save()
    writer.close()
    
# Main Function
plt.close('all')

data_set = 'MET-ENK' # string to determine which data set to use
orig_training,orig_testing,dim = csv_open(data_set)  # returns datframes
training, testing, training_stdev,training_mean = normalization(orig_training,orig_testing) 


kernel = 'joint'
pb = 'on'

excel_path = 'C:/cygwin64/home/jaipe/Thesis/cross-validation/Solver_results/'


# create training sets

trials =[[100,8,	10,	0.001],
         [100,8,	10,	0.001],
         [100,8,	4,	0.001],
        [100,8,4,	0.005],
        [200,10,10,0.001],
        [200,10,10,0.001],
        [200,10,10,0.001],
        [200,10,10,0.01],
        [300,10,10,0.001],
        [300,10,8	,0.001],
        [300,8,10,0.005],
        [300,10,10,0.01],
        [400,10,8,0.001],
        [400,8,10,0.005],
        [400,8,10,0.001],
        [400,8,8,	0.01],
        [500,8,10,0.005],
        [500,8,10,0.001],
        [500,8,10,0.01],
        [500,10,10,0.005]]


full_path = excel_path + data_set+'_'+kernel+'_PBC'+pb+'_'+str(len(trials))+'.xlsx'


MET_ENK_yhat_plot = [[100,0.01,	2,	2],
                    [200,0.1,2,	2],
                    [300,0.001,	2.4	,2.4],
                    [400,0.001	,2.4,	2.8],
                    [500,0.001	,2,	2.8]]

# This section of code is used to loop through parameters and 
# compute Solver MSE and to append results to an excel file
results_list = []
for combo in MET_ENK_yhat_plot:
    N = combo[0]
    C = 4
    epsilon = combo[1]
    g0 = combo[2]
    g1 = combo[2]
    
    X_train,Y_train = training[0:N,0:dim],training[0:N,dim]
    X_test,Y_test= testing[0:N,0:dim],testing[0:N,dim]


    print('------------------')
    soln = Solver(N,orig_training,orig_testing,X_train,Y_train,X_test,Y_test,kernel,training_stdev,training_mean,\
                  C=C,epsilon=epsilon,gamma=1,gamma_vector = [g0,g1], sigma = 1,p =1,pb= pb)   
    MSE,L2 = soln.minimize(data_set,obj ='epsilon_sensitive',FES = 'off',Yhat = 'on',dim_dist = 'off',alpha_dist = 'off')
    results_list.append([N,pb,C,epsilon,g0,g1,MSE,L2,soln.is_psd2()])
    print('------------------')

print(results_list)
#append_solver_results_to_excel(results_list,full_path,kernel)





#N = 100
## ----- Useful for creating PD regions 
##print('-----------------')
#for pb in ['off']:
#    print('For N = ', N)
#    X_train,Y_train = training[0:N,0:dim],training[0:N,dim]
#    X_test,Y_test= testing[0:N,0:dim],testing[0:N,dim]
#    g0_lst = [0.001,0.01]#,0.1,0.2,0.6,1,1.2,1.4,1.6,2,4,6,8,10]
#    g1_lst = [0.001,0.01]#,0.1,0.2,0.6,1,1.2,1.4,1.6,2,4,6,8,10]
#    epsilon_lst = [0.001]#,0.01,0.1,1]
#    C_param = 10
#    #C_lst = [0.1,1]#,2,4,6,8,10] # for rbf
#    #create3d_rbf(N,orig_training,orig_testing,X_train,Y_train,X_test,Y_test,kernel,pb,training_stdev,training_mean,g_lst,epsilon_lst,C_lst)
#    create3d_joint(N,orig_training,orig_testing,X_train,Y_train,X_test,Y_test,kernel,pb,training_stdev,training_mean,g0_lst,g1_lst,epsilon_lst,C_param)


    
    




