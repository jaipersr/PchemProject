
import numpy as np



class objective_class():
    def __init__(self,K_matrix,Y_train,C =1,epsilon = 1,sigma = 1, p =1):
        self.K_matrix = K_matrix
        self.Y_train = Y_train
        self.C = C
        self.epsilon = epsilon
        self.sigma = sigma
        self.p = p
        
    
    def epsilon_sensitive(self,alpha_total):
            alpha,alpha_s = np.split(alpha_total, 2)
            one_vector = np.ones((alpha.shape[0],1))      
            return 0.5*np.dot( np.dot( (alpha - alpha_s).T, self.K_matrix ) , (alpha - alpha_s) ) + self.epsilon*np.dot(one_vector.T, (alpha + alpha_s)) - np.dot(self.Y_train.T,(alpha - alpha_s))
    
    def epsilon_sensitive_grad(self,alpha_total):
        alpha,alpha_s = np.split(alpha_total, 2)
        dual_coef = alpha - alpha_s
        one_vector = np.ones_like(dual_coef)
        A = self.Y_train
        jac1 = np.add(np.add(np.matmul(self.K_matrix,dual_coef),self.epsilon*one_vector),-A)
        jac2 = np.add(np.add(np.matmul(self.K_matrix,-dual_coef),self.epsilon*one_vector),A)
        jac = np.vstack((jac1,jac2))
        jac = np.ndarray.flatten(jac)
        return jac

    # gaussian sensitive objective function
    def gaussian(self,alpha_total):
        alpha,alpha_s = np.split(alpha_total, 2)
        return 0.5*np.dot( np.dot( (alpha - alpha_s).T, self.K_matrix ) , (alpha - alpha_s) ) - np.dot(self.Y_train.T,(alpha - alpha_s)) +0.5*(1/self.C)*(np.dot(alpha.T,alpha)+np.dot(alpha_s.T,alpha))
    
    # Define the jacobian matrix for the first objective function
    def gaussian_grad(self,alpha_total):
        alpha,alpha_s = np.split(alpha_total, 2)
        dual_coef = alpha - alpha_s
        A = self.Y_train
        jac1 = np.add(np.add(np.matmul(self.K_matrix,dual_coef),-A),(1/self.C)*alpha)
        jac2 = np.add(np.add(np.matmul(self.K_matrix,-dual_coef),A),(1/self.C)*alpha_s)
        jac = np.vstack((jac1,jac2))
        jac = np.ndarray.flatten(jac)
        return jac   

    # Huber's robust loss sensitive objective function
    def hubert(self,alpha_total):
        alpha,alpha_s = np.split(alpha_total, 2)
        return 0.5*np.dot( np.dot( (alpha - alpha_s).T, self.K_matrix ) , (alpha - alpha_s) ) - np.dot(self.Y_train.T,(alpha - alpha_s)) +0.5*self.sigma*(1/self.C)*(np.dot(alpha.T,alpha)+np.dot(alpha_s.T,alpha))
        
    
    def hubert_grad(self,alpha_total):
        alpha,alpha_s = np.split(alpha_total, 2)
        dual_coef = alpha - alpha_s
        A = self.Y_train
        jac1 = np.add(np.add(np.matmul(self.K_matrix,dual_coef),-A),(self.sigma/self.C)*alpha)
        jac2 = np.add(np.add(np.matmul(self.K_matrix,-dual_coef),A),(self.sigma/self.C)*alpha_s)
        jac = np.vstack((jac1,jac2))
        jac = np.ndarray.flatten(jac)
        return jac 

    # polynomial sensitive objective function
    def polynomial(self,alpha_total):
        alpha,alpha_s = np.split(alpha_total, 2)
        one_vector = np.ones((alpha.shape[0],1))
        return 0.5*np.dot( np.dot( (alpha - alpha_s).T, self.K_matrix ) , (alpha - alpha_s) ) - np.dot(self.Y_train.T,(alpha - alpha_s)) +((self.p-1)/self.p)*(self.C**-(1/(self.p-1)))*np.dot(one_vector.T,(np.power(alpha,self.p/(self.p-1))+np.power(alpha_s,self.p/(self.p-1))))  

    def polynomial_grad(self,alpha_total):
        alpha,alpha_s = np.split(alpha_total, 2)
        dual_coef = alpha - alpha_s
        A = self.Y_train
        jac1 = np.add(np.add(np.matmul(self.K_matrix,dual_coef),-A),(self.C**(1/(1-self.p)))*np.power(alpha,1/(self.p-1)))
        jac2 = np.add(np.add(np.matmul(self.K_matrix,-dual_coef),A),(self.C**(1/(1-self.p)))*np.power(alpha_s,1/(self.p-1)))
        jac = np.vstack((jac1,jac2))
        jac = np.ndarray.flatten(jac)
        return jac 
    
    # piecewise polynomial sensitive objective function
    def piecewise(alpha_total):
        alpha,alpha_s = np.split(alpha_total, 2)
        one_vector = np.ones((alpha.shape[0],1))
        return 0.5*np.dot( np.dot( (alpha - alpha_s).T, self.K_matrix ) , (alpha - alpha_s) ) - np.dot(self.Y_train.T,(alpha - alpha_s)) +((self.p-1)/self.p)*self.sigma*(C**-(1/(self.p-1)))*np.dot(one_vector.T,(np.power(alpha,self.p/(self.p-1))+np.power(alpha_s,self.p/(self.p-1))))  

    def piecewise_grad(self,alpha_total):
        alpha,alpha_s = np.split(alpha_total, 2)
        dual_coef = alpha - alpha_s
        A = self.Y_train
        jac1 = np.add(np.add(np.matmul(self.K_matrix,dual_coef),-A),self.sigma*(self.C**(1/(1-self.p)))*np.power(alpha,1/(self.p-1)))
        jac2 = np.add(np.add(np.matmul(self.K_matrix,-dual_coef),A),self.sigma*(self.C**(1/(1-self.p)))*np.power(alpha_s,1/(self.p-1)))
        jac = np.vstack((jac1,jac2))
        jac = np.ndarray.flatten(jac)
        return jac 
    
    def get_function(self,obj = 'epsilon_sensitive'):
        
        if obj == 'epsilon_sensitive':
            return self.epsilon_sensitive
        elif obj == 'gaussian':
            return self.gaussian
        elif obj == 'hubert':
            return self.hubert
        elif obj == 'gaussian':
            return self.polynomial
        elif obj == 'piecewise':
            return self.piecewise
        
    def get_grad(self,obj = 'epsilon_sensitive'):
        
        if obj == 'epsilon_sensitive':
            return self.epsilon_sensitive_grad
        elif obj == 'gaussian':
            return self.gaussian_grad
        elif obj == 'hubert':
            return self.hubert_grad
        elif obj == 'gaussian':
            return self.polynomial_grad
        elif obj == 'piecewise':
            return self.piecewise_grad

