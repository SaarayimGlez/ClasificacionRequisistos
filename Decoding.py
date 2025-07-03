# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:12:59 2020

@author: user
"""

from torch import nn
import math
import torch

def conv_out_size(W, K):
    padding=1
    stride=1
    return math.floor((W + 2*padding - K)/stride) + 1

def pool_out_size(W, K):
    stride=2 
    padding=0
    return max(1, math.floor((W + 2*padding - K) / stride) + 1)

def decoding(encoding, num_clases):
    n_conv = encoding[0]
    n_full = encoding[1]
    first_level = encoding[2]
    second_level = encoding[3]

    features = []
    classifier = []
    in_channels = 1
    out_size = 148  
    prev = -1
    pos = 0
    o_sizes = []

    for i in range(n_conv):
        layer = first_level[i]
        n_filters = layer['nfilters']
        f_size = layer['fsize']
        pad = 1
        if f_size > out_size:
            f_size = max(1, out_size - 1)

        if i == 0 or i == 1:
            connections = []
        else:
            connections = second_level[pos:pos+prev]
            for c in range(len(connections)):
                if connections[c] == 1:
                    in_channels += o_sizes[c][1]

        if layer['pool'] == 'off':
            out_size_tmp = conv_out_size(out_size, f_size)
            if out_size_tmp < 1:
                raise ValueError(f"Capa {i}: Tamaño inválido después de conv: {out_size_tmp}")
            operation = [
                nn.Conv1d(in_channels, n_filters, f_size, padding=pad),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(inplace=True)
            ]
            out_size = out_size_tmp

        elif layer['pool'] in ['avg', 'max']:
            p_size = layer['psize']
            
            if f_size > out_size:
                f_size = max(1, out_size - 1)

            out_size_tmp = conv_out_size(out_size, f_size)
            

            if out_size_tmp < 2:
                pool = None 
                out_size = out_size_tmp
                operation = [
                    nn.Conv1d(in_channels, n_filters, f_size, padding=pad),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(inplace=True)
                ]
            else:
                if p_size > out_size_tmp:
                    p_size = max(1, out_size_tmp - 1)
                out_size_tmp = pool_out_size(out_size_tmp, p_size)
                if out_size_tmp < 1:
                    out_size_tmp = 1
                    p_size = 1  
                    
                pool = nn.AvgPool1d(p_size, stride=2) if layer['pool'] == 'avg' else nn.MaxPool1d(p_size, stride=2)
                operation = [
                    nn.Conv1d(in_channels, n_filters, f_size, padding=pad),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(inplace=True),
                    pool
                ]
                out_size = out_size_tmp


        else:
            raise ValueError(f"Capa {i}: tipo de pooling desconocido: {layer['pool']}")

        features.append(operation)
        o_sizes.append([out_size, n_filters])
        in_channels = n_filters
        pos += prev if i > 1 else 0
        prev += 1

    in_size = out_size * in_channels
    if in_size <= 0:
        raise ValueError(f"Error: tamaño de entrada inválido para capa lineal: {in_size}")

    for i in range(n_conv, n_conv + n_full):
        layer = first_level[i]
        n_neurons = layer['neurons']
        classifier.append(nn.Linear(in_size, n_neurons))
        classifier.append(nn.ReLU(inplace=True))
        in_size = n_neurons

    classifier.append(nn.Linear(in_size, num_clases))
    return features, classifier, o_sizes


'''Networks class'''
class CNN(nn.Module):
  def __init__(self, encoding, features, classifier, sizes, init_weights = True):
    super(CNN, self).__init__()
    extraction = []
    for layer in features:
      extraction += layer
    self.extraction = nn.Sequential(*extraction)
    self.classifier = nn.Sequential(*classifier)
    self.features = features
    self.second_level = encoding[3]
    self.sizes = sizes
    
  def forward(self, x):
    '''Feature extraction'''
    prev = -1
    pos = 0
    outputs = {}
    features = self.features
    #print(x.shape)
    for i in range(len(features)):
      #print('Layer: ', i)
      if i == 0 or i == 1:
        x = nn.Sequential(*features[i])(x)
        outputs[i] = x
        #print(x.shape)
      
      else:
        connections = self.second_level[pos:pos+prev]
        for c in range(len(connections)):
          if connections[c] == 1:
            skip_size = self.sizes[c][0] 
            req_size = x.shape[2] 
            #print('X: ',x.shape)
            if skip_size > req_size:
              psize = skip_size - req_size + 1 
              pool = nn.MaxPool1d(kernel_size = psize, stride = 1) 
              x2 = pool(outputs[c])
            if skip_size == req_size:
              x2 = outputs[c]
            if req_size == skip_size + 1:
              pool = nn.MaxPool1d(kernel_size = 2, stride = 1, padding = (1,1))
              x2 = pool(outputs[c])
            if req_size == skip_size + 2:
              pad = int((req_size - skip_size)/2)
              padding = nn.ConstantPad1d((pad, pad), 0)
              x2 = padding(outputs[c])
            #print('X2: ',x2.shape)
            x = torch.cat((x, x2), axis = 1)
          
        x = nn.Sequential(*features[i])(x)
        #print('Out size: ', x.shape)
        outputs[i] = x
        pos += prev
      
      prev += 1
    
    #print('Classification size: ', x.shape)
    x = torch.flatten(x,1)
    '''Classification'''
    '''for l in self.classifier:
      x = l(x)'''
    x = self.classifier(x)
    #print(x.shape)
    return nn.functional.log_softmax(x, dim=1)