
from __future__ import division
import numpy as np
import pandas as pd
import scipy.sparse as sp
dot = np.dot

class pca:
    """
    Compute a k-dimensional embedding of matrix mat using PCA
    Arguments:
    df: a two-dimensional pandas DataFrame with columns userid, itemid and rating
    axis: which axis to treat as examples (0 or 1)
    k: number of dimensions in embedding
    Returns:
    numpy matrix of shape (m,k) where m is the number of unique userids in df when axis=0
    and m is the number of unique itemids in df when axis=1
    """
    def __init__(self, df, axis=0, k=2):
        
        self.rating = df['rating']
        
        self.A = sp.csr_matrix( (df['rating'],(df['userid'],df['itemid'])), shape=(len(df['userid'][indx].values), len(df['itemid'][indx].values)) )

        if axis == 1:
            self.A = self.A.transpose()
        #center the data
        self.mean = self.A.mean()
        self.A -= self.mean
        #scale the data
        std = self.A.std()
        self.std = np.where( std, std, 1. )
        if verbose:
            print "Center /= A.std:", self.std
        self.A /= self.std #end scaling
        #end centering

        #
        self.U, self.d, self.Vt = sp.linalg.svd( self.A )#leaving out optional 'K' so that we can get however many we want from doing the calculations again on stored data.
        assert np.all( self.d[:-1] >= self.d[1:] )  # sorted
        self.eigen = self.d**2
        self.sumvariance = np.cumsum(self.eigen)
        self.sumvariance /= self.sumvariance[-1]
        #npc -> number of principal components that are greater than .90 of variance
        self.npc = np.searchsorted( self.sumvariance, .90 ) + 1
        
        self.dinv = np.array([ 1/d if d > self.d[0] * 1e-6  else 0
                                for d in self.d ])
        
        self.ret = self.U[:, :k] * self.d[:k]

