### Class to perform PCA analysis for UNCC BINF6210 Machine Learning class.
### Jennifer Gilby, Fall 2023

import numpy as np
import pandas as pd


class PCA:
    
    def __init__(self,num_components): # initialize Pca object 
        self.meanCentered = None
        self.p_components = None # .transpose() is loadings  
        self.num_components = num_components 
        self.eigenvalues = None
        self.covMat = None
    
    def fit(self, data): # fits data to model (gives P) 
       
       # mean centering
        self.mean = np.mean(data,axis=0)
        self.meanCentered = data - self.mean
        
        # covariance matrix 
        covarianceMatrix = np.cov(self.meanCentered.astype(float), rowvar=False) # or .transpose() 
        self.covMat = covarianceMatrix
       
        # eignen-decomposition
        self.eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)
            # can do .eigh() to just get real values
        
        # sort eigenvectors 
        eigenvectors = eigenvectors.transpose() 
        indices = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[indices]
        eigenvectors = eigenvectors[indices]

        # store first n eigenvectors/principal components
        self.p_components = eigenvectors[0:self.num_components]
    
    
    def transform(self, data):  # project data onto principal component axes (gives Y, which is XP)
        return np.dot(self.meanCentered, self.p_components.transpose()) # call this for scores 

    
    def explained_var(self):
        self.eigenvalues = list(self.eigenvalues)
        explained_variance = []
        
        # sort eigenvalues in descending order 
        self.eigenvalues.sort(reverse=True)

        # calculate explained variance 
        eigenvalues_total = sum(self.eigenvalues)
        explained_variance = [(x/eigenvalues_total)*100 for x in self.eigenvalues]
        explained_variance = np.round(explained_variance, 2)
        
        return explained_variance

    
    def variance(self):
        return list(self.eigenvalues)
