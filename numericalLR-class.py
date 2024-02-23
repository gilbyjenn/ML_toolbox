### Class to perform Numerical Linear Regression (gradient descent) analysis for UNCC BINF6210 Machine Learning class.
### Jennifer Gilby, Fall 2023

import numpy as np
import pandas as pd

class numericalLR:
    
    def __init__(self, alpha, theta_0, theta_1, convergence):
        
        # user input parameters 
        self.alpha = alpha
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.convergence = convergence 
        
        # data 
        self.X = None
        self.Y = None
        
        self.diff = 100
        self.yhat = None
        self.xbar = None
        self.xbar = None
        self.MSreg = None
        self.MSerr = None
        self.R2 = None
        self.yhat = None


    
    def standardize(self, data):   
       
        mean = np.mean(data,axis=0)
        std = np.std(data,axis=0)
        
        self.X = data[:,0]
        self.Y = data[:,1]
     
        self.ybar = mean[1] 
        self.xbar = mean[0]
        
        
        return self.X, self.Y
    
    
    def h(self, X, theta_0, theta_1):
        
        return self.theta_0 + (self.theta_1 * self.X)
    

    
    def J(self, theta_0, theta_1, X, Y):
            
        self.diff = self.h(self.X, self.theta_0, self.theta_1) - (self.Y)
        diff_squared = self.diff**2
            
        return (diff_squared.sum()/(len(self.X)*2))
    

    
    def gradient(self, theta_0, theta_1, alpha, X, Y):
        
        t_0 = ((self.h(self.X, self.theta_0, self.theta_1) - (self.Y)))
        t_1 = ((t_0)*self.X)

    
        temp_theta_0 = (self.alpha / len(self.X)) * t_0.sum()
        temp_theta_1 = (self.alpha / len(self.X)) * t_1.sum()
    
        new_theta_0 = self.theta_0 - temp_theta_0
        new_theta_1 = self.theta_1 - temp_theta_1
    
        return (new_theta_0, new_theta_1)
    
    
    def fit_predict(self, data):
        
        self.X, self.Y = self.standardize(data)
        
        while self.diff >= convergence: 
            
            initial_cost = self.J(self.theta_0, self.theta_1, self.X, self.Y)
            self.theta_0, self.theta_1 = self.gradient(self.theta_0, self.theta_1, self.alpha, self.X, self.Y)
            new_cost = self.J(self.theta_0, self.theta_1, self.X, self.Y)
            self.diff = initial_cost - new_cost 
            
    
    def regression_line(self):
        
        self.yhat = self.theta_0 + self.theta_1*self.X
        
        return self.yhat 
    
    
    
    def score(self):
        
        self.yhat = self.regression_line()
        
        # calculate SStot = sum(y - ybar)^2
        SStot = 0
        for i in range(0, len(self.Y)):
            SStot += ((self.Y[i]-self.ybar)**2)

        # calculate SSreg = sum(yhat - ybar)^2 
        SSreg = 0
        for i in range(0, len(self.Y)):
            SSreg += ((self.yhat[i] - self.ybar)**2)

        # calculate SSerr = sum(ri)^2 = sum(yi - B0 - B1*xi)^2
        SSerr = 0 
        for i in range(0, len(self.Y)):
            SSerr += ((self.Y[i] - self.theta_0 - self.theta_1*self.X[i])**2)

        # sum of squares error 
        # calculate MStot 
        MStot = SStot/(len(self.Y)-1)

        # calculate MSreg 
        self.MSreg = SSreg  

        # calculate MSerr 
        self.MSerr = SSerr/(len(self.Y)-2)

        # calculate R2
        self.R2 = SSreg/SStot 

        return self.R2
    
    
    def getStats(self):
        
        # calculate F statistic 
        F = self.MSreg/self.MSerr  
        
        # calculate p-value
        dfd = len((self.Y)*len(self.X) - len(self.X))
        dfn = len(self.Y) - 1
        
        p = f.sf(F, dfn, dfd)
        
        
        return F, p