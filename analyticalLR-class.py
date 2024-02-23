### Class to perform Analytical Linear Regression analysis for UNCC BINF6210 Machine Learning class.
### Jennifer Gilby, Fall 2023

import numpy as np
import pandas as pd

class analyticalLR:
    
    def __init__(self): 
        
        self.standardized = None 
        self.X = None
        self.Y = None 
        self.yhat = None 
        self.B0 = None
        self.B1 = None
        self.ybar = None
        self.xbar = None
        self.MSreg = None
        self.MSerr = None
        self.F = None
        self.p = None
         
    
    def fit(self, data):   

        mean = np.mean(data,axis=0)
        std = np.std(data,axis=0)

        self.X = data[:,0]
        self.Y = data[:,1]
    
        self.xbar = mean[0] 
        self.ybar = mean[1]

        
        # calculate b1 hat = covariance (X,Y) / variance(X)
        covMatrix = np.cov(data, rowvar=False) 
        cov = float(covMatrix[1:, :1]) 
        
        var = float(covMatrix[0,0])
  
        self.B1 = cov/var
        
        # calculate b0 = ybar - B1*xbar
        self.B0 = self.ybar - self.B1 * self.xbar

            
        # regression line / y hat values 
        self.yhat = self.B0 + self.B1*self.X       
    
    
    def predict(self):
        return self.yhat
        

    def score(self): # calculate and return correlation coeffecient 
        
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
            SSerr += ((self.Y[i] - self.B0 - self.B1*self.X[i])**2)
            
        # sum of squares error 
        # calculate MStot 
        MStot = SStot/(len(self.Y)-1)
        
        # calculate MSreg 
        self.MSreg = SSreg  
        
        # calculate MSerr 
        self.MSerr = SSerr/(len(self.Y)-2)
        
        # calculate R2
        R2 = SSreg/SStot 
        
        return R2
    
    
    def getStats(self):
        
        # calculate F statistic 
        self.F = self.MSreg/self.MSerr  
        
        # calculate p-value
        dfd = len((self.Y)*len(self.X) - len(self.X))
        dfn = len(self.Y) - 1
        
        self.p = f.sf(self.F, dfn, dfd)
        
        
        return self.F, self.p