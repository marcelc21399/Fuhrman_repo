#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 23:40:09 2019

@author: marcello
"""
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
plt.close("all")


import numpy as np

import pickle
import sys
def ld_obj(s):
    if '.pkl' not in s:
        s = s + '.pkl'
    
    with open(s, 'rb') as f:
        if sys.version_info.major > 2:
            x = pickle.load(f, encoding='latin1')#bytes latin1
        else:
            x = pickle.load(f)
        return x
def stretch(a, new_len):
    return np.interp(np.linspace(0,len(a),new_len), np.arange(len(a)), a)
def my_scatter(ar):
    plt.scatter(np.arange(len(ar)),ar)
plt.close('all')

s='eval_F_ISUP_Comb_out/history.pkl'

hist = ld_obj(s)

L=len(hist["lr"])
xs = np.arange(L)

used=np.argmin(hist["val_loss"])
print(used)
print(hist["val_loss"][used])
print(hist["loss"][used])


plt.close('all')
v = plt.scatter(xs, hist["val_loss"], 20, 'tab:orange')
t = plt.scatter(xs, hist["loss"], 20, 'b')
plt.legend((t,v),('loss', 'val_loss'))
plt.xlabel('Epoch')
plt.ylabel('Loss (Cross-Entropy)')
plt.title('Model Loss over Epoch')
#plt.show()
plt.savefig('Loss_Over_Epoch_All', format='eps', dpi = 1000)
plt.ylim((0.2,1))
plt.savefig('Loss_Over_Epoch_y_lim', format='eps', dpi = 1000)


plt.close('all')
v = plt.scatter(xs, hist["val_acc"], 20, 'tab:orange')
t = plt.scatter(xs, hist["acc"], 20, 'b')
plt.legend((t,v),('acc', 'val_acc'))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy over Epoch')
#plt.show()
plt.savefig('Accuracy_Over_Epoch_All', format='eps', dpi = 1000)