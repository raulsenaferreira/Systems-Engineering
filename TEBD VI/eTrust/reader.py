# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:26:25 2015

@author: ygorc
"""

import scipy.io as sio

ratings = sio.loadmat("rating_with_timestamp.mat")["rating"]
trust = sio.loadmat("epinion_trust_with_timestamp.mat")["trust"]
