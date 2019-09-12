#ld_obj, save_obj, bothIms_and_Demo, perfMeasures, weighted_categorical_crossentropy, hist_type
from keras.layers import Concatenate, Input, Dense, Reshape, Dropout, Lambda, Flatten
from keras import regularizers
from keras.models import Model

"pickle helpers"
"______________"

import pickle
import sys

from time import gmtime, strftime

import numpy as np
from keras import backend as K

import pandas

def ld_obj(s):
    if '.pkl' not in s:
        s = s + '.pkl'
    
    with open(s, 'rb') as f:
        if sys.version_info.major > 2:
            x = pickle.load(f, encoding='latin1')#bytes latin1
        else:
            x = pickle.load(f)
        return x
    
def save_obj(obj, s ):
    with open(s + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol = 2)

        
#yy = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )( xx )
def bothIms_and_Demo(model1, model2, dr = 0.5, nb_classes = 2, kr = 0.1, ar = 0.1, nd = 10):
    #model1 = ResNet50(include_top=True, weights='imagenet', classes=nb_classes, input_shape = (224,224,7,1))
    #model2 = ResNet50(include_top=True, weights='imagenet', classes=nb_classes, input_shape = (224,224,7,1))
    
    for l in model1.layers:
        l.name = l.name+'_1'
    for l in model2.layers:
        l.name = l.name+'_2'
        
    #l = model1.layers.pop()
    x1 = model1.layers[-3].output
    #x1 = Reshape((2048,))(x1)
    x1 = Flatten()(x1)
    
    x2 = model2.layers[-3].output
    #x2 = Reshape((2048,))(x2)
    x2 = Flatten()(x2)
    

    dense_1 = Concatenate(axis = -1)([x1, x2])
    #dense_1a = Reshape((4096,))(dense_1)
    
    #tmp = Flatten()(dense_1)
    #dense_1 = Dropout(dr)(dense_1)
    #tf.reset_default_graph()
    dense_2 = Dense(1024,)(dense_1)
    #dense_2 = Dropout(dr)(dense_2)
    dense_3 = Dense(24,)(dense_2)
    
    demo_inp = Input(shape=(nd,))
    
    dense_4 = Concatenate(axis = -1)([dense_3, demo_inp])
    
    x = Dropout(dr)(dense_4)##
    
    dense = Dense(nb_classes, name='predictions', kernel_regularizer=regularizers.l2(kr), activity_regularizer=regularizers.l1(ar))(x)
    
    dense_soft = Lambda(lambda x: K.softmax(x))(dense)
    

    
    
    full_model = Model(inputs = [model1.input, model2.input, demo_inp], outputs = [dense_soft])
    return full_model

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


"functions for performance evaluation"
"____________________________________"
# evaluate performance metrics for one threshold
def perfIter(p1, th, dic, y1, yz):
    h1 = p1 >= th
    hz = np.logical_not(h1)

    TN = float(np.sum(np.logical_and(hz,yz)))
    FN = float(np.sum(np.logical_and(hz,y1)))
    FP = float(np.sum(np.logical_and(h1,yz)))
    TP = float(np.sum(np.logical_and(h1,y1)))

    if TP+FN != 0:
        TPR = TP/(TP+FN)
    else:
        TPR = -1

    if TN+FP != 0:
        TNR = TN/(TN+FP)
    else:
        TNR = -1

    if TP+FP != 0:
        Precision = TP/(TP+FP)
    else:
        Precision = -1

    AM = (TPR+TNR)/2
    GM = np.sqrt(TPR*TNR)

    if TPR + Precision != 0:
        F1 = 2 * TPR * Precision / (TPR + Precision)
    else:
        F1 = -1

    FPR = 1 - TNR

    dic['TN'].append(TN)
    dic['FN'].append(FN)
    dic['FP'].append(FP)
    dic['TP'].append(TP)
    dic['AM'].append(AM)
    dic['GM'].append(GM)
    dic['F1'].append(F1)
    dic['Precision'].append(Precision)
    dic['Recall'].append(TPR)
    dic['TPR'].append(TPR)
    dic['FPR'].append(FPR)
    dic['Acc'].append(np.mean(h1 == y1))
    return dic

# initatialize dictionary entry with lists
def initDic():
    dic = {}
    dic['TN'] = []
    dic['FN'] = []
    dic['FP'] = []
    dic['TP'] = []
    dic['AM'] = []
    dic['GM'] = []
    dic['F1'] = []
    dic['Precision'] = []
    dic['Recall'] = []
    dic['TPR'] = []
    dic['FPR'] = []
    dic['Acc'] = []
    return dic

# takes in probability vector and labels and makes GM, AM, F1, TPR, FPRm, ROC, PR metrics along with intermediates
def perfMeasures(p1, n, lbl, nm = strftime("%H_%M_%S", gmtime())+'/'):
    

  p1 = p1[:,1]
  thresholds = np.linspace(np.min(p1),np.max(p1),n)
  
  print(nm)
  y1 = np.argmax(lbl, axis = 1)
  yz = np.logical_not(y1)

  RL, PL = [], []
  dic = initDic()
  
  for th in thresholds:
      dic = perfIter(p1, th, dic, y1, yz)

  FPRL = dic['FPR']
  TPRL = dic['TPR']
  RL, PL = dic['Recall'], dic['Precision']
  
  Tar = np.array(TPRL)
  Far = np.array(FPRL)
  plt.figure()
  plt.plot(Far,Tar)
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.title("ROC")
  plt.savefig("../out/"+nm+"ROC.png")
  AUCROC = -np.trapz(Tar, x=Far)

  Tar = np.array(PL)
  Far = np.array(RL)
  plt.figure()
  plt.plot(Far,Tar)
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title("Precision-Recall")
  plt.savefig("../out/"+nm+"Precision.png")
  AUCPR = -np.trapz(Tar, x=Far)
  
  dic_act = initDic()
  dic_act = perfIter(p1, 0.5, dic_act, y1, yz)

  dic_act['AUCPR'] = AUCPR
  dic_act['AUCROC'] = AUCROC
  return dic, dic_act


def weighted_categorical_crossentropy(weights):
    """m
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss



def hist_type(ar):
    for elt in np.unique(ar):
        print(str((elt))+':'+str(np.sum(ar==elt)))
    out = np.zeros((len(np.unique(ar)),))
    for i,elt in enumerate(np.unique(ar)):
        print(str((elt))+':'+str(np.mean(ar==elt)))
        out[i] = np.mean(ar==elt)
    return out
def restrict(csv_name, oldIms, oldLs):
    oldData = np.hstack((oldIms, np.expand_dims(oldLs, axis=1)))
    
    data = pandas.read_csv(csv_name)
    nm = list(data.iloc[:,0])
    
    newData = []
    remainder = []
    for elt in oldData:
        if elt[3] in nm:
            print(elt[3])
            newData.append(elt)
        else:
            remainder.append(elt)
    newData = np.array(newData)
    newIms, newLs = newData[:,:4], newData[:,4]
    return newIms, newLs, remainder