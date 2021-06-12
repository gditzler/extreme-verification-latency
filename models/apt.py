#!/usr/bin/env python 

import numpy as np 
import scipy as sp

from sklearn.cluster import KMeans


class APT(): 
    def __init__(self, 
                 classifier, 
                 Xinit,
                 Yinit,
                 Kclusters,   
                 T, 
                 method): 
        """
        """
        # total number of times we are going to run an experiment with .run()
        self.T = T 
        # number of unique classes in the data 
        self.nclasses = len(np.unique(Yinit))
        # set the intial data and clusters 
        self.Xinit = Xinit
        self.Yinit = Yinit 
        self.Kclusers = Kclusters
        self.class_cluster = np.zeros((self.nclasses,))
        self.__initialize()
        self.M = len(Yinit)
        
    def __initialize(self): 
        """
        """
        # run the clustering algorithm on the training data then find the cluster 
        # assignment for each of the samples in the training data 
        self.cluster = KMeans(n_clusters=self.Kclusers).fit(self.Xinit)
        labels = self.class_cluster.predict(self.Xinit)

        # for each of the clusters, find the labels of the data samples in the clusters
        # then look at the labels from the initially labeled data that are in the same
        # cluster. assign the cluster the label of the most frequent class. 
        for i in range(self.Kclusers): 
            yhat = self.Yinit[labels==i]
            self.class_cluster[i] = sp.stats.mode(yhat)


    
    def run(self, Xts, Yts, Uts): 
        """
        """
        

        for t in range(self.T): 
            Xt = Xts[t]
            Yt = Yts[t]
            Ut = Uts[t]
            N = len(Xt)

            # check lens of the data 
            if self.M != N: 
                raise ValueError('N and M must be the same size')
            
            # step 4: associate each new instance to one previous example
            sample_assignment = np.zeros((N,))
            for n in range(N): 
                sample_assignment[n] = np.argmin(np.linalg(Xt[n] - self.Xinit, axis=1))

            # step 5: Compute instance-to-exemplar correspondence
            # step 6: Pass the cluster assignment from the example to their 
            # assigned instances to achieve instance-to-cluster assignment 
            # step 7: pass the class of an example  