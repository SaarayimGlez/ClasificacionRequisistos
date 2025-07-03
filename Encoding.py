# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:57:16 2020

@author: Gustavo Vargas Hakim
"""

import random

class Encoding:
    
    def __init__(self):
        '''Hyperparameters configuration'''
        #Convolutional layers
        self.FSIZES = [3,5,7]
        self.NFILTERS = [4,8,16]
    
        #Pooling layers
        self.PSIZES = [2,3,4,5]
        self.PTYPE = ['max', 'avg']
    
        #Fully connected layers
        self.NEURONS = [4,8,16,32,64,128]
    
    
    
    
    """
        Creates the list of layers of the first level of a neural network
        
        Params:
                minC: minimum number of convolutional layers
                macC: maximum number of convolutional layers
                minF: minimum number of fully-connected layers
                maxF: maximum number of fully-connected layers
                n_conv: actual number of convolutional layers
                n_full: actual number of fully-connected layers
    
        return: list of individual parameter dictionaries for each layer
    
    """
    def first_level_encoding(self, minC, maxC, minF, maxF, n_conv, n_full):

        first_level = []
        
        for i in range(n_conv):
            layer = {'type' : 'conv',
                     'nfilters' : random.choice(self.NFILTERS),
                     'fsize' : random.choice(self.FSIZES),
                     'pool' : random.choice(['max', 'avg', 'off']),
                     'psize' : random.choice(self.PSIZES)
                    }
            first_level.append(layer)
        
        #Fully connected part
        for i in range(n_full):
            layer = {'type' : 'fc',
                     'neurons' : random.choice(self.NEURONS)}
            
            first_level.append(layer)
            
        return first_level
            
    
    
    """
        Second level encoding that generates a random binary encoding 
        that represents residual connections
        
        Params: 
            n_conv: number of convolutional layers
            
        return: binary list that defines the skip connections between 
                convolutional layers
    
    """
    def second_level_encoding(self, n_conv):
        second_level = []
        prev = -1
        for i in range(n_conv):
            if prev < 1:
                prev += 1
            if prev >= 1:
                for _ in range(prev-1):
                    second_level.append(random.choice([0,1]))
                prev += 1
                
        return second_level
    
    
    
            
    def encoding(self, minC, maxC, minF, maxF):
        n_conv = random.randint(minC, maxC)
        n_full = random.randint(minF, maxF)
        
        first_level = self.first_level_encoding(minC, maxC, minF, maxF, n_conv, n_full)
        second_level = self.second_level_encoding(n_conv)
        
        return n_conv, n_full, first_level, second_level
        
                
            
                
            
        