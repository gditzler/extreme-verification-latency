#!/usr/bin/env python 

# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import numpy as np 
import scipy as sp
import tqdm

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
        

        for t in tqdm(self.T): 
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
            self.cluster = KMeans(Xt, init=self.class_cluster.cluster_centers_)
            # step 7: pass the class of an example  