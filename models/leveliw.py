#!/usr/bin/env python 



class LevelIW(): 
    def __init__(self, 
                 classifier, 
                 T, 
                 method): 
        """
        """
        self.classifier = classifier 
        self.T = T
    
    def run(self, Xt, Yt, Ut): 
        """
        """
        self.classifier