# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:53:10 2018

@author: jrslagle
"""

class Hyper(object):
    
    def __init__(self, *args, **kwargs):
        self.sizes = kwargs.get('sizes',None)
        self.seed = kwargs.get('seed',None)
        self.epochs = kwargs.get('epochs',None)
        self.batch_size = kwargs.get('batch_size',None)
        self.learning_rate = kwargs.get('learning_rate',None)
        self.score = kwargs.get('score',None)