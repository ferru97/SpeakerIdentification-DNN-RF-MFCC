# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:56:15 2019

@author: Vito
"""

import numpy as np

x = np.array([[1, 2], [3,4]])

x2 = np.array([[5,6], [7,8],[[5,6], [7,8]]])

z = np.concatenate((x,x2), axis = 1)


