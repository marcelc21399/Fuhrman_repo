import matplotlib

matplotlib.use('Agg')

import numpy as np
from keras import backend as K
from keras.utils import np_utils
import pandas
import os
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import nrrd
from time import gmtime, strftime
from scipy.ndimage.interpolation import rotate
import re



import sys
sys.path.insert(0, 'keras-resnet-master')
from resnet import ResnetBuilder

from keras.models import Model
import numpy as np
import warnings

from keras.models import Sequential 
from keras.layers import Input
from keras import layers
from keras.layers import Dense, Reshape
from keras.layers import Activation, LeakyReLU, PReLU
from keras.layers import Flatten
from keras.layers import Conv2D, Conv3D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D, Concatenate
from keras.layers import AveragePooling2D, Reshape, Lambda
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization, Dropout, GaussianNoise
from keras import regularizers
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
import csv

from skimage.transform import resize


import my_globs

import numpy as np

import os
from time import gmtime, strftime

import sys
sys.path.insert(0, 'keras-resnet-master')
from resnet import ResnetBuilder

import tensorflow as tf

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import np_utils
K.clear_session()


def readCSVs(csv_path):
    data = pandas.read_csv(csv_path)
    nms = data.iloc[:,0].values
    
    inds = np.array([i for i,elt in enumerate(nms) if hasNumbers(elt)])
    
    nms = nms[inds]
    #1=low, 2=high
    pathols = data.iloc[inds,1].values
    
    #1=Fuhrman, 2=ISUP
    Fuhr_ISUP = data.iloc[inds,2].values
    
    #1=biopsy, 0=not    
    biopsy = [elt == 'biopsy' for elt in data.iloc[inds,3].values]
    
    ages = data.iloc[inds,4].values
    
    genders = data.iloc[inds,5].values
    
    demos = np.vstack([ages, genders, Fuhr_ISUP, biopsy, pathols])
    
    return list(nms), np.transpose(demos)

def collectInCSV(imsLis, folLis, valsLis, nm, demo, segLis):
    collect = []
    for fol, ims, vals, seg in zip(folLis, imsLis, valsLis, segLis):
        if fol in nm:
            i = nm.index(fol)
            row = [fol, ims, vals, demo[i], seg]
            collect.append(row)
        else:
            print(fol)
    return collect

from Ianto_preprocess import preprocess_pack

def my_crop(image, seg, dic):
    image_cropped = image.copy()
    
    # image_cropped[~seg.astype(bool)] = 0 # rmv background, can comment out
    
    bb = np.argwhere(seg)
    (x0, y0, z0) = bb.min(0)
    (x1, y1, z1) = bb.max(0) + 1
    
    ls = []
    for elt in dic['space directions']:
        l = np.sum(np.square(np.array([float(n) for n in elt])))
        ls.append(l)
    dims = np.multiply(ls, dic['sizes'])
    
    vol = np.prod(ls) * np.sum(seg)
    
    image_cropped = image_cropped[x0:x1,y0:y1,z0:z1]
    
    return image_cropped, np.append(dims, vol) #output volume, dims
def getImsSegs(imsF, seq):
    tum, _ = nrrd.read(imsF + '/'+seq+'/imagingVolume.nrrd')
    seg,  seg_dic = nrrd.read(imsF + '/'+seq+'/segMask_tumor.seg.nrrd')
    
    # create extended mask
    seg_full = np.zeros(tum.shape)
    uni_off = seg_dic['keyvaluepairs']['Segmentation_ReferenceImageExtentOffset']
    off = list(map(int,str(uni_off).split()))
    x0,y0,z0 = off
    x1,y1,z1 = list(map(lambda x,y: x+y, off, list(seg.shape)))
    
    seg_full[x0:x1,y0:y1,z0:z1] = seg
    return tum.astype(np.uint32), seg_full.astype(np.uint8), seg_dic, seg.astype(bool)
def parseDir(dr):
    RCC = dirNoDots(dr)
    
    lis = []
    nmLis = []
    
    RCC = [elt for elt in RCC if ('.' not in elt and '\r' not in elt)]
    RCC = [elt for elt in RCC if hasNumbers(elt)]
    RCC = [elt for elt in RCC if 'idney' in elt or 'ideny' in elt]
    RCC = [elt for elt in RCC if '_' not in elt]
    
    valsLis = []
    folLis = []
    imsLis = []
    segLis = []
    #f=RCC[0]
    for f in RCC:
        imsF = dr + '/' + f
        print(imsF)
        
        folLis.append(f)
        
        tum_t1c, seg_t1c, dic_t1c, seg_cropped_t1c = getImsSegs(imsF, 'T1C')
        tum_t2, seg_t2, dic_t2, seg_cropped_t2 = getImsSegs(imsF, 'T2WI')
        
        #reference, images_w_segmentations=tum_t1c, [(tum_t2, seg_t2)]
        out_t1_image, out_t2 = preprocess_pack(tum_t1c, [(tum_t2, seg_t2)], use_n4_bias=False, use_registration=False)
        out_t1_seg = seg_t1c #unchanged
        out_t2_image, out_t2_seg = out_t2[0]
        
        cropped_t1, vals_t1c = my_crop(out_t1_image, out_t1_seg, dic_t1c)
        cropped_t2, vals_t2 = my_crop(out_t2_image, out_t2_seg, dic_t2)
        
        ims = [cropped_t1, cropped_t2]
        imsLis.append(ims)
        segLis.append([seg_cropped_t1c, seg_cropped_t2])
        
        vals = [vals_t1c, vals_t2]
        valsLis.append(vals)
    return valsLis, imsLis, folLis, segLis

def readData():
    nm, demo = readCSVs('../RCC_histological_grade.csv')
    dr = '../grade'
    valsLis, imsLis, folLis, segLis = parseDir(dr)
    
    collect = collectInCSV(imsLis, folLis, valsLis, nm, demo, segLis)
    
    #collect row:
    #[str, [T1_im,T2_im], [(4,),(4,)], (5,)]
    #(4,): x,y,z,vol
    #(5,): ages, genders, Fuhr_ISUP, biopsy, pathols
    #1=Fuhrman, 2=ISUP, 
    
    st = np.array(collect, dtype=object)
    return st

def saveRawData():
    st = readData()
    save_obj(st, '../inps')

def loadRawData():
    st = ld_obj('../inps') 
    return st


def getLargestAlongDim(im, seg, dim, w):
	#im, dim = seq, 2
	dims = [0,1,2]
	dims = np.setdiff1d(dims, dim)
	slice_sizes = np.apply_over_axes(np.sum, seg, dims)
	largest_slice_ind = np.argmax(slice_sizes)
	largest_slice = np.take(im, largest_slice_ind, axis=dim)
	rszd = resize(largest_slice, (w, w))
	return rszd

#STOCHASTIC
def mk_trn_tst_val_inds(by_patient, trn_tst_val):
    # by_patient, trn_tst_val= non_biopsy_non_Fuhrman, spl
    #[input_T1s[:,:,:,i], input_T2s[:,:,:,i], input_numbers[i,:], nms[i], outs[i]] for i in range(len(outputs))]
    by_patient = np.array(by_patient, dtype=object)
    inps = by_patient[:, :4]
    outs = by_patient[:, 4]
    
    
    outs = np.array(outs)
    uniq = np.unique(outs)
    tst_inds = []
    trn_inds = []
    val_inds = []

    for elt in uniq:
        inds_elt = np.where(outs==elt)[0]
        
        l = len(inds_elt)
        inds = np.arange(l)
        np.random.shuffle(inds)

        cutoff_trn = int(np.ceil(l*trn_tst_val[0]))
        cutoff_tst = int(np.ceil(l*trn_tst_val[1])) + cutoff_trn
        #cutoff_val = int(np.ceil(l*trn_tst_val[2])) + cutoff_tst

        
        trn_inds.append(inds_elt[inds[:cutoff_trn]])
        tst_inds.append(inds_elt[inds[cutoff_trn:cutoff_tst]])
        val_inds.append(inds_elt[inds[cutoff_tst:]])
    
    tst_inds = np.hstack(tst_inds)
    trn_inds = np.hstack(trn_inds)
    val_inds = np.hstack(val_inds)
    
    save_obj([trn_inds, val_inds, tst_inds],'inds')
    
def split_trn_tst_val(by_patient):
    by_patient = np.array(by_patient, dtype=object)
    inps = by_patient[:, :4]
    outs = by_patient[:, 4]-1
    outs = outs.astype(bool)
    
    
    [trn_inds, val_inds, tst_inds] = ld_obj('inds')
    
    tstLs = outs[tst_inds]
    trnLs = outs[trn_inds]
    valLs = outs[val_inds]
    
    tstIms = inps[tst_inds]
    trnIms = inps[trn_inds]
    valIms = inps[val_inds]
    return trnLs, trnIms, tstLs, tstIms, valLs, valIms
    

def save_inps():
    w, h = 64, 3
    st = loadRawData()
    
    nms = list(st[:,0])
    
    raw_ims = list(st[:,1])
    
    dims = list(st[:,2])
    
    demos = np.vstack(st[:,3])
    ages, genders, Fuhr_ISUP, biopsy, pathols = demos[:,0], demos[:,1], demos[:,2], demos[:,3], demos[:,4]
    
    raw_segs = list(st[:,4])
    
    inp_ims = np.zeros((len(raw_ims),2), dtype=object)
    for i, (ims, segs) in enumerate(zip(raw_ims, raw_segs)):
        for j, (im, seg) in enumerate(zip(ims, segs)):
            plane_slices = [getLargestAlongDim(im, seg, dim, w) for dim in [0, 1, 2]]
            
            inp_ims[i,j] = np.dstack(plane_slices)
    input_T1s = np.stack(inp_ims[:,0], axis=3)
    input_T2s = np.stack(inp_ims[:,1], axis=3)
    
    input_dims = np.vstack([np.hstack(elt) for elt in dims])
    input_demos = np.transpose(np.vstack([ages, genders]))
    input_numbers = np.hstack([input_dims, input_demos])
    
    outs = pathols
    
    by_patient = [[input_T1s[:,:,:,i], input_T2s[:,:,:,i], input_numbers[i,:], nms[i], outs[i]] for i in range(len(outputs))]
    
    non_biopsy = biopsy == 0
    Fuhrman = Fuhr_ISUP == 1
    included = np.logical_and(non_biopsy, Fuhrman)
    
    
    non_biopsy_non_Fuhrman = [by_patient[elt] for elt in np.where(included)[0]]
    
    #[str, [T1_im,T2_im], [(4,),(4,)], (5,)]
    #(4,): x,y,z,vol
    #(5,): ages, genders, Fuhr_ISUP, biopsy, pathols
    #1=Fuhrman, 2=ISUP
    
    
    spl = [0.7, 0.1, 0.2]
    #mk_trn_tst_val_inds(by_patient, spl)
    
    trnLs, trnIms, tstLs, tstIms, valLs, valIms = split_trn_tst_val(non_biopsy_non_Fuhrman)
    

    save_obj([trnLs, trnIms, tstLs, tstIms, valLs, valIms], '../inps_64_orthogonal')
    
    biopsy_pats = np.array([by_patient[elt] for elt in np.where(biopsy)[0]], dtype=object)
    ISUP_pats = np.array([by_patient[elt] for elt in np.where(Fuhr_ISUP == 2)[0]], dtype=object)
    save_obj([biopsy_pats, ISUP_pats], '../misc_64_orthogonal')












































# randomly split to left+right
def randSplit(diff):
    if diff > 0:
        r = np.random.randint(diff)
        l = diff - r
    else:
        l,r = 0,0
    return l, r

def randCrop(im, desired):
    #im, desired =tmp0, wid
    # if smaller pad to 224,224 randomly, make sure identical changes to mask, im
    [h, w, _] = im.shape
    if h < desired:
        l, r = randSplit(desired - h)
        im = np.pad(im, ((l, r), (0, 0),(0, 0)), 'constant')
    if w < desired:
        l, r = randSplit(desired - w)
        im = np.pad(im, ((0, 0), (l, r),(0, 0)), 'constant')
    return im
def midCrop(im, desired):
    #im, desired =tmp0, wid
    # if smaller pad to 224,224 randomly, make sure identical changes to mask, im
    [h, w, _] = im.shape
    if h < desired:
        half = (desired - h)/2.0
        l, r = int(np.floor(half)), int(np.ceil(half))
        im = np.pad(im, ((l, r), (0, 0),(0, 0)), 'constant')
    if w < desired:
        half = (desired - w)/2.0
        l, r = int(np.floor(half)), int(np.ceil(half))
        im = np.pad(im, ((0, 0), (l, r),(0, 0)), 'constant')
    return im

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

"functions for visualization"
"___________________________"
# isolates slices on 2nd axis and plots each slice
def plotSlices(toPlt):
    t = toPlt.shape[2]
    fig, axes = plt.subplots(1, t)
    if t == 1:
        axes.imshow(toPlt[:,:,0])
    else:
        for x in range(t):
            axes[x].imshow(toPlt[:,:,x])

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

"initatialize some variables for debugging convenience"
"_____________________________________________________"

drs = drs = ['../../segment_Penn_renal_tumor']
dr = drs[0]
test_perc = 0.1
stride = 2
n_degs = 7
cols = 1
nb_classes = 2
wid, hi = 40, 2


"functions for reading inputs"
"____________________________"

# helper finds all files except . or ..
def dirNoDots(nm):
    ls = os.listdir(nm)
    return [f for f in ls if not f.startswith('.')]

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


"functions for enriching/processing read inputs"
"______________________________________________"

def randomShuffle(ls, n):
    #n = len(label)
    temp = np.arange(n)
    np.random.shuffle(temp)
    newL = ls[temp,:]
    return newL

# helper to roll images in list by z units along ax axis
def rollLs(list_of_ims, z, ax):
    newL = []
    for im in list_of_ims:
        newL.append(np.roll(im, z, axis=ax))
    return newL

# pad to (v,v,c) with zeros
def padTo(ar, v, c):
    # ar, v, c=tmp0, wid, hi
    [h, w, d] = ar.shape
    m = np.mean(ar)
    
    if v-h>=0 and c - d>=0:
        out =  np.pad(ar, ((0, v - h), (0, v - w), (0, c - d)), 'constant')
    else: 
        out =  np.zeros((80,80,10))
    return out

# measure bbox
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

#this version does uniform xy resize and expands separately along z then pads
def rsz(mat, wid):
    [a, b, c] = mat.shape
    wid_f = float(wid)
    #hi_f = float(hi)
    sh = [a, b, c]
    ab = min(wid_f/a, wid_f/b)
    tmp0 = zoom(mat, (ab, ab, 1))
    tmp = padTo(tmp0, wid, c)
    return tmp, sh

def rsz3D(mat, wid, hi):
    #mat, wid, hi = row[1][T1C[0]][1], 224, 7
    # mat, wid, hi = im, w, hi
    [a, b, c] = mat.shape
    wid_f = float(wid)
    hi_f = float(hi)
    sh = [a, b, c]
    ab = min(wid_f/a, wid_f/b)
    tmp0 = zoom(mat, (ab, ab, hi_f/c))
    tmp = midCrop(tmp0, wid)#tmp = padTo(tmp0, wid, hi)
    # plotSlices(tmp)
    return tmp, sh

def rszUP(mat, wid_f, wid):
    [a, b, c] = mat.shape
    ab = min(wid_f/a, wid_f/b)
    tmp0 = zoom(mat, (ab, ab, 1))
    tmp = padTo(tmp0, wid, c)
    return tmp

# helper to rot images
def my_rot(im, ang):
    # im, ang = im, d
    non_bin = rotate(im, ang, axes=(1, 0), reshape=False, mode='constant', prefilter=True, cval = np.min(im))
    binIm = non_bin
    #binIm[non_bin<0]=0
    # binIm[non_bin>=0.5]=1 # used in binary images
    return binIm

# takes in Lists of raw segmentation images and enriches by x,y,z transforms and rotations
def enrichTransformExt(newtr, stride, n_degs):
    newtr_orig = newtr
    rs = []
    for row in np.transpose(newtr_orig):
        x = row[0]
        u = row[1]
        s = x.shape[0]
        tot1 = np.sum(x, axis = 2) > 0
        tot2 = np.sum(u, axis = 2) > 0
        tot = (tot1 + tot2) > 0
        bbox = bbox2(tot)
        r0 = np.setdiff1d(range(-bbox[0],s-bbox[1]-1,1),[0])
        r1 = np.setdiff1d(range(-bbox[2],s-bbox[3]-1,1),[0])
        if len(r0)+len(r1)>4:
            r0 = np.setdiff1d(range(-bbox[0],s-bbox[1]-1,stride),[0])
            r1 = np.setdiff1d(range(-bbox[2],s-bbox[3]-1,stride),[0])
        rs.append(len(r0) + len(r1))
    mx = max(rs)
    for row in np.transpose(newtr_orig):
        x = row[0]
        u = row[1]
        s = x.shape[0]
        tot1 = np.sum(x, axis = 2) > 0
        tot2 = np.sum(u, axis = 2) > 0
        tot = (tot1 + tot2) > 0
        bbox = bbox2(tot)
        def appShifts(ax, rng, T1T2struct):
            for x_shift in rng:
                row[0] = np.roll(x, x_shift, axis=ax)
                row[1] = rollLs(u, x_shift, ax)
                T1T2struct = np.column_stack((T1T2struct, row))
            return T1T2struct
        r0 = np.setdiff1d(range(-bbox[0],s-bbox[1]-1,1),[0])
        r1 = np.setdiff1d(range(-bbox[2],s-bbox[3]-1,1),[0])
        if len(r0)+len(r1)>4:
            r0 = np.setdiff1d(range(-bbox[0],s-bbox[1]-1,2),[0])
            r1 = np.setdiff1d(range(-bbox[2],s-bbox[3]-1,2),[0])
        newtr = appShifts(0, r0, newtr)
        newtr = appShifts(1, r1, newtr)
    #TwoDs, unPadded, szs, TwoDLbls, demoL = T1T2struct.tolist()
    
    # rotation enrichment
    degs = np.setdiff1d(np.linspace(0,360,n_degs),[0,360])
    for row in np.transpose(newtr_orig):
        x = row[0]
        u = row[1]
        for d in degs:
            row[0] = my_rot(x, d)
            row[1] = list(map(lambda x: my_rot(x,d),u))
            newtr = np.column_stack((newtr, row))
    return newtr
    
# takes in Lists of raw segmentation images and enriches by x,y,z transforms and rotations
def enrichTransformRot(T1T2struct, stride, n_degs):
    T1T2struct_orig = T1T2struct
    rs = []
    for row in np.transpose(T1T2struct_orig):
        x = row[0]
        u = row[1]
        s = row[2]
        tot1 = np.sum(x, axis = 2) > 0
        if isinstance(u, list):
            tots_u = []
            for elt in u:
                tots_u.append(np.sum(elt, axis = 2) > 0)
            tot2 = sum(tots_u)
        else:
            tot2 = np.sum(u, axis = 2) > 0
        tot = tot1+tot2>0
        bbox = bbox2(tot)
        r0 = np.setdiff1d(range(-bbox[0],s-bbox[1]-1,1),[0])
        r1 = np.setdiff1d(range(-bbox[2],s-bbox[3]-1,1),[0])
        if len(r0)+len(r1)>4:
            r0 = np.setdiff1d(range(-bbox[0],s-bbox[1]-1,stride),[0])
            r1 = np.setdiff1d(range(-bbox[2],s-bbox[3]-1,stride),[0])
        rs.append(len(r0) + len(r1))
    mx = max(rs)
    for row in np.transpose(T1T2struct_orig):
        x = row[0]
        u = row[1]
        s = row[2]
        tot1 = np.sum(x, axis = 2) > 0
        tot2 = np.sum(u[0], axis = 2) > 0
        tot3 = np.sum(u[1], axis = 2) > 0
        tot = (tot1+tot2+tot3) > 0
        bbox = bbox2(tot)
        def appShifts(ax, rng, T1T2struct):
            for x_shift in rng:
                row[0] = np.roll(x, x_shift, axis=ax)
                row[1] = rollLs(u, x_shift, ax)
                T1T2struct = np.column_stack((T1T2struct, row))
            return T1T2struct
        r0 = np.setdiff1d(range(-bbox[0],s-bbox[1]-1,1),[0])
        r1 = np.setdiff1d(range(-bbox[2],s-bbox[3]-1,1),[0])
        if len(r0)+len(r1)>4:
            r0 = np.setdiff1d(range(-bbox[0],s-bbox[1]-1,2),[0])
            r1 = np.setdiff1d(range(-bbox[2],s-bbox[3]-1,2),[0])
        T1T2struct = appShifts(0, r0, T1T2struct)
        T1T2struct = appShifts(1, r1, T1T2struct)
    #TwoDs, unPadded, szs, TwoDLbls, demoL = T1T2struct.tolist()
    
    # rotation enrichment
    degs = np.setdiff1d(np.linspace(0,360,n_degs),[0,360])
    for row in np.transpose(T1T2struct_orig):
        x = row[0]
        u = row[1]
        for d in degs:
            row[0] = my_rot(x, d)
            row[1] = list(map(lambda x: my_rot(x,d),u))
            T1T2struct = np.column_stack((T1T2struct, row))
    return T1T2struct
#TwoDs, unPadded, szs, TwoDLbls, demoL = T1T2struct.tolist()
# def flipSag(x):
#     return np.flip(x, axis = 2)
# 1:    np.fliplr, 2: np.flipud, 3: flipSag
def flip_12(x):
    return np.fliplr(np.flipud(x))
# def flip_13(x):
#     return np.fliplr(flipSag(x))
# def flip_23(x):
#     return np.flipud(flipSag(x))
# def flip_123(x):
#     return np.fliplr(np.flipud(flipSag(x)))
def flippy(T1T2struct_t, fun):
    row0 = T1T2struct_t[0,:]
    row1 = T1T2struct_t[1,:]

    row0_1 = list(map(fun, row0.tolist()))
    row1_1 = list(map(lambda x: list(map(fun, x)), row1.tolist()))
    
    outT1T2struct_t = T1T2struct_t.tolist()
    
    outT1T2struct_t[0] = row0_1
    outT1T2struct_t[1] = row1_1
    return np.array(outT1T2struct_t)
def flippyExt(T1T2struct_t, fun):
    row0 = T1T2struct_t[0,:]
    row1 = T1T2struct_t[1,:]

    row0_1 = list(map(fun, row0.tolist()))
    row1_1 = list(map(fun, row1.tolist()))
    
    outT1T2struct_t = T1T2struct_t.tolist()
    
    outT1T2struct_t[0] = row0_1
    outT1T2struct_t[1] = row1_1
    return np.array(outT1T2struct_t)


def flipEnrichExt(T1T2struct_i):
    T1T2structLR = flippyExt(T1T2struct_i, np.fliplr)
    T1T2structUD = flippyExt(T1T2struct_i, np.flipud)
    T1T2struct_LR_UD = flippyExt(T1T2struct_i, flip_12)
    return np.column_stack((T1T2struct_i, T1T2structLR, T1T2structUD, T1T2struct_LR_UD))
def flipEnrich(T1T2struct_i):


    
    T1T2structLR = flippy(T1T2struct_i, np.fliplr)
    T1T2structUD = flippy(T1T2struct_i, np.flipud)
    T1T2struct_LR_UD = flippy(T1T2struct_i, flip_12)
    
    return np.column_stack((T1T2struct_i, T1T2structLR, T1T2structUD, T1T2struct_LR_UD))
def collectNmsImsTyps(drs):
    nmLis = []
    imLis = []
    types = []
    for i, dr in enumerate(drs):
        nmL, imL = parseDir(dr)
        tp = np.full((len(nmL),), i).tolist()
        nmLis = nmLis + nmL
        imLis = imLis + imL
        types = types + tp
    out = np.column_stack((nmLis, imLis, types))
    return out
csv_path = '../results.csv'
# lists of all segs and names
drs = ['../../segment_Penn_renal_tumor']

    
def readInps(wid, hi, cols, nm, y, demo, drs):
    '''
    # read names-labels
    data = pandas.read_csv(csv_path)
    nm = data['ID'].values
    y = data['pathology(1=rcc,2=oncocytoma)'].values - 1
    demo = data.loc[: , "age":"location=(upper=1, interpolate=2 lower=3)"].values
    #??figure out the other features, maybe mean substract and do standard variance??
    '''
    nb_classes = len(np.unique(y))
    
    # collect all data unmatched
    objAr = collectNmsImsTyps(drs)
    oobjAr = objAr
    #objAr = oobjAr
    ###
    [nmLis, imLis, types] = np.transpose(objAr).tolist()
    
    z_szs = np.array([im.shape[2] for im in imLis])
    
    objAr=objAr[z_szs>2,:]
    
    [nmLis, imLis, types] = np.transpose(objAr).tolist()
    
    nimLis = []
    for im in imLis:
        sl_szs = np.sum(np.sum(im>0, axis = 0),axis=0)
        top2 = np.argsort(sl_szs)[-3:]
        nimLis.append(im[:,:,top2])
    objAr = np.column_stack((nmLis, nimLis, types))
    #
    ###
    
    # get rid of rows where we don't have label
    out = []
    for i, row in enumerate(objAr):
        [n, im, tp] = row
        f_s = n[:n.find('/')]
        if '-' in f_s:
            f_s = f_s[:-2]
        num = f_s
        #print(np.sum(num == nm))
        if np.sum(num == nm) >= 1:
            row = np.append(row, num)
            if len(out) != 0:
                out = np.column_stack((out, row))
            else:
                out = row
            
    [nmLis, imLis, types, nums] = out

    wid_f = float(wid)
    hi_f = float(hi)
    shp = (wid,wid,hi)
    def resizedSeqIms(st, hasNum):
        T2 = [i for i,tmp_nm in enumerate(nmLis[hasNum]) if st in tmp_nm]
        T2Ar = np.zeros(shp)
        orig_T2Ar = T2Ar
        s2 = [0,0,0]
        d = -1
        if len(T2) == 1:
            orig_T2Ar = imLis[hasNum[T2[0]]]
            T2Ar, s2 = rsz(orig_T2Ar, wid_f, wid, hi_f, hi)
            d = types[hasNum[T2[0]]]
        unPT2 = rszUP(orig_T2Ar, wid_f, wid)
        return T2Ar, s2, unPT2, d
    
    
    TwoDs = []
    TwoDLbls = []
    demoL = []
    
    
    #stack the T2 and T1C if they exist
    szs = []
    nmLis = np.array(nmLis)
    unPadded = []
    
    T1T2struct = []
    for num in set(nums):
        hasNum = np.where(num == nums)[0]
        T2Ar, s2, unPT2, d2 = resizedSeqIms('T2WI', hasNum)
        T1CAr, s1C, unPT1C, d1C = resizedSeqIms('T1C', hasNum)
        if np.sum(T2Ar)==0 or np.sum(T1CAr[:])==0:
            continue
            j=0
            
        d = max(d2, d1C)
        if d==-1:
            print('AHHHH')
            
        im_const_d = np.dstack([T1CAr, T2Ar])
        im_var_d = [unPT1C, unPT2]
        
        TwoDs.append(im_const_d)
        unPadded.append(im_var_d)
        
        szs.append(wid)
        
        ind_num = num == nm
        
        demo_slice = demo[ind_num]
        
        TwoDLbls.append(y[ind_num][0])
        demo_s = np.array(list(demo_slice[0])+s2+s1C)#+[d])
        demoL.append(demo_s)
        ###
        row = [im_const_d, im_var_d, wid, y[ind_num][0], demo_s]
        if len(T1T2struct) != 0:
            T1T2struct = np.column_stack((T1T2struct, row))
        else:
            T1T2struct = row
    setS = set(szs)
    
    

    
    
    #enrich with tranformations
    if transRot:
        T1T2struct = enrichTransformRot(T1T2struct, stride, n_degs)
    TwoDs, TwoDLbls, szs, demoL, unPadded = T1T2struct
    
    structs = []
    for sz in setS:
        inds = szs == sz
        T1T2struct_i = T1T2struct[:,inds]

        # if flip set then enrich by flipping
        if flip:
            T1T2struct_i = flipEnrich(T1T2struct_i)
            
        Ts, unPadded, szs, TwoDLbls, demoL = T1T2struct_i.tolist()
        
        #Ts = T1T2struct_i[0,:].tolist()
        Ts = list(map(lambda x: np.expand_dims(x,-1), Ts))
        T1 = list(map(lambda x: x[:,:,:hi,:], Ts))
        T2 = list(map(lambda x: x[:,:,hi:,:], Ts))
        
        struct = np.array([T1,T2,unPadded, szs, TwoDLbls, demoL])
        structs.append(struct)
    return structs


def getBatches(t1_up, t2_up):
    sss = np.zeros((2,len(t2_up)))
    for t1, t2, i in zip(t1_up, t2_up, range(len(t2_up))):
        sss[0,i]=t1.shape[2]
        sss[1,i]=t2.shape[2]
    collectBatches = []
    i = 0
    S = set()
    for i in range(len(t1_up)):
        if i in S:
    
            nothing=0# print('dunzo')        
        else:
            inds = np.logical_and(sss[0,i]==sss[0,:], sss[1,i]==sss[1,:])
            wh = np.where(inds)[0]
            
            S.update(wh.tolist())
            collectBatches.append(wh)
    return collectBatches

# returns the inputs to the models (use function to reduce mem pressure)


# returns the inputs to the models (use function to reduce mem pressure)
def getMdlInps(test_perc, nm, demo, y, drs = ['../Final original plus segmentations RCC vs. oncytoma for RSNA abstract/RCC-seg/', '../Final original plus segmentations RCC vs. oncytoma for RSNA abstract/Oncocytoma-seg/']):
    # 1 color (keras requires last dim of input to be color ie 1, 3, etc)
    cols = 1
    # number of classes (benign = 0, malignant = 1)
    
    # all images interpolated/padded to 40x40x5 since most images around or under this size
    wid = 224
    hi = 3

    structs = readInps(wid, hi, cols, nm, y, demo, drs)
    #struct = np.array([T1,T2,unPadded, szs, TwoDLbls, demoL])
    struct = structs[0]
        
    T1C,T2,unPadded, szs, label, demo = struct.tolist()
    #T1C, T2, label, demo = map(lambda y: map(lambda x: np.array(x,dtype=np.float32),y),[T1C,T2, label, demo])
    
    t1_up = list(map(lambda x: x[0].astype(np.float32), unPadded))
    t2_up = list(map(lambda x: x[1].astype(np.float32), unPadded))
    
    obj = np.array([T1C,T2,t1_up,t2_up, szs, label, demo])
    #convrt to lighter datatype for rearrange?
    
    "set datatype"
    obj_shuff = np.transpose(randomShuffle(np.transpose(obj), len(label)))
    
    
    n = int(obj_shuff.shape[1]*test_perc)
    ts = obj_shuff
    T1C,T2,t1_up,t2_up, _, label, demo = ts.tolist()
    T1C = np.squeeze(T1C)
    T2 = np.squeeze(T2)
    T1C, T2, demo, label = list(map(lambda x: np.array(x,dtype=np.float32),[T1C, T2, demo, label]))
    label = np_utils.to_categorical(label, nb_classes)
    ts = [T1C, T2, demo, label, t1_up, t2_up]
    return ts

def shuffleReorg(data, test_perc):
    obj_shuff = np.transpose(randomShuffle(np.transpose(data), data.shape[1]))
    n = int(obj_shuff.shape[1]*test_perc)
    ts = obj_shuff[:,:n]
    tr = obj_shuff[:,n:]
    
    
    T1C, T2, demo, label, t1_up, t2_up = ts.tolist()
    T1C = np.squeeze(T1C)
    T2 = np.squeeze(T2)
    T1C, T2, demo, label = list(map(lambda x: np.array(x,dtype=np.float32),[T1C, T2, demo, label]))
    ts = [T1C, T2, demo, label, t1_up, t2_up]
    
    
    T1C, T2, demo, label, t1_up, t2_up = tr.tolist()
    T1C = np.squeeze(T1C)
    T2 = np.squeeze(T2)
    T1C, T2, demo, label = list(map(lambda x: np.array(x,dtype=np.float32),[T1C, T2, demo, label]))
    tr = [T1C, T2, demo, label, t1_up, t2_up]
    return tr, ts

"pickle helpers"
"______________"

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
    
def save_obj(obj, s ):
    with open(s + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol = 2)
'''
st1 = st[:200,:]
st2 = st[200:,:]
#st3 = np.vstack([st1,st2])
save_obj(st1, 'st1') 
save_obj(st2, 'st2') 
st2 = ld_obj('st') 
'''
#save_obj(st, '../st') 
#st2 = ld_obj('../st') 
        


def save_inps():
    w, h = 64, 3
    st = loadSt()
    Topt = np.array([1,2])
    inps, outs, demos = getFurman(st, Topt)
    ims, demos = compileImsDemos(inps, demos, w, h)
    rsz_inps = stack_obj_ar(ims)
    rsz_inps = [[im,d] for im, d in zip(rsz_inps, demos)]
    spl = [0.7, 0.1, 0.2]
    trnLs, trnIms, tstLs, tstIms, valLs, valIms = split_trn_tst_val(rsz_inps, outs, spl)
    save_obj([trnLs, trnIms, tstLs, tstIms, valLs, valIms], '../data/inps')
def save_inps_rcc():
    w, h = 64, 3
    st = loadSt()
    Topt = np.array([1,2])
    inps, outs, demos = getFurman(st, Topt)
    ims, demos = compileImsDemos(inps, demos, w, h)
    rsz_inps = stack_obj_ar(ims)
    rsz_inps = [[im,d] for im, d in zip(rsz_inps, demos)]
    spl = [0.7, 0.1, 0.2]
    trnLs, trnIms, valLs, valIms = split_trn_tst_val_rcc(rsz_inps, outs, spl)
    save_obj([trnLs, trnIms, valLs, valIms], '../data/inps_RCC')
        
        
def save_st():
    st = loadSt2();
    st1 = st[:200,:]
    st2 = st[200:,:]
    save_obj(st1, '../data/st1') 
    save_obj(st2, '../data/st2') 
def save_new_data():
    drs = ['Penn_new_case']
    nm, demo, Fuhrman_vs_ISUP = readCSVs_Fuhrman_ISUP('../clinical_data_renal_tumor2.csv')
    nm = list(nm)
    drs = ['../../' + elt for elt in drs]
    imsLis, folLis, dicLis = [], [], []
    for dr in drs:
        dics, _, dirIms, dirFols = parseDir(dr)
        imsLis = imsLis + dirIms
        folLis = folLis + dirFols
        dicLis = dicLis + dics
    
    checkHasT1T2(imsLis, folLis)
    collect = collectInCSV(dicLis, folLis, nm, demo)
    
    tmp_szs = []
    for ims in imsLis:
        for im in ims:
            tmp_szs.append(im[1].shape)
            
    zs = [elt[2] for elt in tmp_szs]
    newStruct = []
    for row in collect:
        T1Im = -1
        T2Im = -1
        T2I = [i for i,elt in enumerate(row[1]) if elt[0]=='T2WI']
        T1C = [i for i,elt in enumerate(row[1]) if elt[0]=='T1C']
        if T1C:
            T1Im = row[1][T1C[0]][1:3]
            #T1Im, s = rsz3D(row[1][T1C[0]][1], 224, 7)
            #T1Im = getLargestSlice_Resize(row[1][T1C[0]][1], (224, 224))
        if T2I:
            T2Im = row[1][T2I[0]][1:3]
            # T2Im, s = rsz3D(row[1][T2I[0]][1], 224, 7)
            #T2Im = getLargestSlice_Resize(row[1][T2I[0]][1], (224, 224))
        newStruct.append([row[0], row[2], T1Im, T2Im])
    st = np.array(newStruct)
    #save_obj(st, '../inp_new_penn')
    
    new_st = []
    for row in st:
        print(Fuhrman_vs_ISUP[nm.index(row[0])])
        new_st.append(np.append(row, Fuhrman_vs_ISUP[nm.index(row[0])]))
    new_st = np.array(new_st)
    #save_obj(new_st, '../inp_new_penn_Fuhrman_vs_ISUP')
    
    st_Fuhrman = []
    st_ISUP = []
    for row in st:
        print(Fuhrman_vs_ISUP[nm.index(row[0])])
        if Fuhrman_vs_ISUP[nm.index(row[0])]==1:
            st_Fuhrman.append(row)
        else:
            st_ISUP.append(row)
    st_Fuhrman = np.array(st_Fuhrman)
    st_ISUP = np.array(st_ISUP)
    #save_obj(st_Fuhrman, '../inp_new_penn_Fuhrman')
    #save_obj(st_ISUP, '../inp_new_penn_ISUP')
        
        
    #st = ld_obj(st, '../inp_new_penn')
    
    
    #rsz_inps, outs = st_to_inp_out(st)
    #save_obj([rsz_inps, outs],'inps_new_data')
    
    rsz_inps, outs = st_to_inp_out(st_Fuhrman)
    save_obj([rsz_inps, outs],'../data/inps_new_data_Fuhrman')
    
    rsz_inps, outs = st_to_inp_out(st_ISUP)
    save_obj([rsz_inps, outs],'../data/inps_new_data_ISUP')
    
    
    
    
    
    
    
    # limit to RCC: RCC = ['1' in elt[1][0] for elt in new_st]
    st_Fuhrman_cc = np.array([st_Fuhrman[elt] for elt in np.where(['1' in elt[1][0] for elt in st_Fuhrman])[0]])
    st_ISUP_cc = np.array([st_ISUP[elt] for elt in np.where(['1' in elt[1][0] for elt in st_ISUP])[0]])
    
        
    
    rsz_inps, outs = st_to_inp_out(st_Fuhrman_cc)
    save_obj([rsz_inps, outs],'../data/inps_new_data_Fuhrman_cc')
    
    rsz_inps, outs = st_to_inp_out(st_ISUP_cc)
    save_obj([rsz_inps, outs],'../data/inps_new_data_ISUP_cc')
        
        
        
        
def resizedSeqIms(st, hasNum):
    T2 = [i for i,tmp_nm in enumerate(nmLis[hasNum]) if st in tmp_nm]
    T2Ar = np.zeros(shp)
    orig_T2Ar = T2Ar
    s2 = [0,0,0]
    d = -1
    if len(T2) == 1:
        orig_T2Ar = imLis[hasNum[T2[0]]]
        T2Ar, s2 = rsz(orig_T2Ar, wid_f, wid, hi_f, hi)
        d = types[hasNum[T2[0]]]
    unPT2 = rszUP(orig_T2Ar, wid_f, wid)
    return T2Ar, s2, unPT2, d
desired= 224

'''
drs = ['Penn_new_case']
'''
def loadSt_new_penn():
    '''
    drs = ['Penn_new_case']
    nm, demo = readCSVs('../clinical_data_renal_tumor.csv')
    nm = list(nm)
    drs = ['../../' + elt for elt in drs]
    imsLis, folLis, dicLis = [], [], []
    for dr in drs:
        dics, _, dirIms, dirFols = parseDir(dr)
        imsLis = imsLis + dirIms
        folLis = folLis + dirFols
        dicLis = dicLis + dics
    
    checkHasT1T2(imsLis, folLis)
    collect = collectInCSV(dicLis, folLis, nm, demo)
    '''
    tmp_szs = []
    for ims in imsLis:
        for im in ims:
            tmp_szs.append(im[1].shape)
    zs = [elt[2] for elt in tmp_szs]
    '''
    newStruct = []
    for row in collect:
        T1Im = -1
        T2Im = -1
        T2I = [i for i,elt in enumerate(row[1]) if elt[0]=='T2WI']
        T1C = [i for i,elt in enumerate(row[1]) if elt[0]=='T1C']
        if T1C:
            T1Im = row[1][T1C[0]][1:3]
            #T1Im, s = rsz3D(row[1][T1C[0]][1], 224, 7)
            #T1Im = getLargestSlice_Resize(row[1][T1C[0]][1], (224, 224))
        if T2I:
            T2Im = row[1][T2I[0]][1:3]
            # T2Im, s = rsz3D(row[1][T2I[0]][1], 224, 7)
            #T2Im = getLargestSlice_Resize(row[1][T2I[0]][1], (224, 224))
        newStruct.append([row[0], row[2], T1Im, T2Im])
    st = np.array(newStruct)
    save_obj(st, '../inp_new_penn')
    '''
    st = ld_obj(st, '../inp_new_penn')
    
    w, h = 64, 3
    Topt = np.array([1,2])
    inps, outs, demos = getFurmanReg(st, Topt)
    outs = outs == 1
    ims, demos = compileImsDemos(inps, demos, w, h)
    rsz_inps = stack_obj_ar(ims)
    rsz_inps = [[im,d] for im, d in zip(rsz_inps, demos)]
    outs = list(outs)
    
    save_obj([rsz_inps, outs],'inps_new_data')
    
    return rsz_inps, outs
def loadSt_new_penn_differentiate_Fuhrman_ISUP():
    '''
    drs = ['Penn_new_case']
    nm, demo, Fuhrman_vs_ISUP = readCSVs_Fuhrman_ISUP('../clinical_data_renal_tumor2.csv')
    nm = list(nm)
    drs = ['../../' + elt for elt in drs]
    imsLis, folLis, dicLis = [], [], []
    for dr in drs:
        dics, _, dirIms, dirFols = parseDir(dr)
        imsLis = imsLis + dirIms
        folLis = folLis + dirFols
        dicLis = dicLis + dics
    
    checkHasT1T2(imsLis, folLis)
    collect = collectInCSV(dicLis, folLis, nm, demo)
    
    #has_RCC = np.array(['1' in elt[2][0] for elt in collect])
    #collect1 = [collect[elt] for elt in np.where(has_RCC)[0]]
    
    '''
    tmp_szs = []
    for ims in imsLis:
        for im in ims:
            tmp_szs.append(im[1].shape)
    zs = [elt[2] for elt in tmp_szs]
    '''
    newStruct = []
    for row in collect:
        T1Im = -1
        T2Im = -1
        T2I = [i for i,elt in enumerate(row[1]) if elt[0]=='T2WI']
        T1C = [i for i,elt in enumerate(row[1]) if elt[0]=='T1C']
        if T1C:
            T1Im = row[1][T1C[0]][1:3]
            #T1Im, s = rsz3D(row[1][T1C[0]][1], 224, 7)
            #T1Im = getLargestSlice_Resize(row[1][T1C[0]][1], (224, 224))
        if T2I:
            T2Im = row[1][T2I[0]][1:3]
            # T2Im, s = rsz3D(row[1][T2I[0]][1], 224, 7)
            #T2Im = getLargestSlice_Resize(row[1][T2I[0]][1], (224, 224))
        newStruct.append([row[0], row[2], T1Im, T2Im])
    st = np.array(newStruct)
    save_obj(st, '../inp_new_penn')
    '''
    new_st = []
    for row in st:
        print(Fuhrman_vs_ISUP[nm.index(row[0])])
        new_st.append(np.append(row, Fuhrman_vs_ISUP[nm.index(row[0])]))
    new_st = np.array(new_st)
    save_obj(new_st, '../inp_new_penn_Fuhrman_vs_ISUP')
    
    st_Fuhrman = []
    st_ISUP = []
    for row in st:
        print(Fuhrman_vs_ISUP[nm.index(row[0])])
        if Fuhrman_vs_ISUP[nm.index(row[0])]==1:
            st_Fuhrman.append(row)
        else:
            st_ISUP.append(row)
    st_Fuhrman = np.array(st_Fuhrman)
    st_ISUP = np.array(st_ISUP)
    save_obj(st_Fuhrman, '../inp_new_penn_Fuhrman')
    save_obj(st_ISUP, '../inp_new_penn_ISUP')
        
        
    st = ld_obj(st, '../inp_new_penn')
    
    
    rsz_inps, outs = st_to_inp_out(st)
    save_obj([rsz_inps, outs],'inps_new_data')
    
    rsz_inps, outs = st_to_inp_out(st_Fuhrman)
    save_obj([rsz_inps, outs],'inps_new_data_Fuhrman')
    
    rsz_inps, outs = st_to_inp_out(st_ISUP)
    save_obj([rsz_inps, outs],'inps_new_data_ISUP')
    
    return rsz_inps, outs
    return rsz_inps, outs
def loadSt_new_penn_differentiate_Fuhrman_ISUP_CC_only():
    '''
    drs = ['Penn_new_case']
    nm, demo, Fuhrman_vs_ISUP = readCSVs_Fuhrman_ISUP('../clinical_data_renal_tumor2.csv')
    nm = list(nm)
    drs = ['../../' + elt for elt in drs]
    imsLis, folLis, dicLis = [], [], []
    for dr in drs:
        dics, _, dirIms, dirFols = parseDir(dr)
        imsLis = imsLis + dirIms
        folLis = folLis + dirFols
        dicLis = dicLis + dics
    
    checkHasT1T2(imsLis, folLis)
    collect = collectInCSV(dicLis, folLis, nm, demo)
    
    #has_RCC = np.array(['1' in elt[2][0] for elt in collect])
    #collect1 = [collect[elt] for elt in np.where(has_RCC)[0]]
    
    '''
    tmp_szs = []
    for ims in imsLis:
        for im in ims:
            tmp_szs.append(im[1].shape)
    zs = [elt[2] for elt in tmp_szs]
    '''
    newStruct = []
    for row in collect:
        T1Im = -1
        T2Im = -1
        T2I = [i for i,elt in enumerate(row[1]) if elt[0]=='T2WI']
        T1C = [i for i,elt in enumerate(row[1]) if elt[0]=='T1C']
        if T1C:
            T1Im = row[1][T1C[0]][1:3]
            #T1Im, s = rsz3D(row[1][T1C[0]][1], 224, 7)
            #T1Im = getLargestSlice_Resize(row[1][T1C[0]][1], (224, 224))
        if T2I:
            T2Im = row[1][T2I[0]][1:3]
            # T2Im, s = rsz3D(row[1][T2I[0]][1], 224, 7)
            #T2Im = getLargestSlice_Resize(row[1][T2I[0]][1], (224, 224))
        newStruct.append([row[0], row[2], T1Im, T2Im])
    st = np.array(newStruct)
    save_obj(st, '../inp_new_penn')
    '''
    new_st = []
    for row in st:
        print(Fuhrman_vs_ISUP[nm.index(row[0])])
        new_st.append(np.append(row, Fuhrman_vs_ISUP[nm.index(row[0])]))
    new_st = np.array(new_st)
    save_obj(new_st, '../inp_new_penn_Fuhrman_vs_ISUP')
    
    st_Fuhrman = []
    st_ISUP = []
    for row in st:
        print(Fuhrman_vs_ISUP[nm.index(row[0])])
        if Fuhrman_vs_ISUP[nm.index(row[0])]==1:
            st_Fuhrman.append(row)
        else:
            st_ISUP.append(row)
    st_Fuhrman = np.array(st_Fuhrman)
    st_ISUP = np.array(st_ISUP)
    
    # limit to RCC: RCC = ['1' in elt[1][0] for elt in new_st]
    st_Fuhrman_cc = np.array([st_Fuhrman[elt] for elt in np.where(['1' in elt[1][0] for elt in st_Fuhrman])[0]])
    st_ISUP_cc = np.array([st_ISUP[elt] for elt in np.where(['1' in elt[1][0] for elt in st_ISUP])[0]])
    save_obj(st_Fuhrman, '../inp_new_penn_Fuhrman')
    save_obj(st_ISUP, '../inp_new_penn_ISUP')
    
        
    st = ld_obj(st, '../inp_new_penn')
    
    
    rsz_inps, outs = st_to_inp_out(st)
    save_obj([rsz_inps, outs],'inps_new_data')
    
    rsz_inps, outs = st_to_inp_out(st_Fuhrman_cc)
    save_obj([rsz_inps, outs],'inps_new_data_Fuhrman_cc')
    
    rsz_inps, outs = st_to_inp_out(st_ISUP_cc)
    save_obj([rsz_inps, outs],'inps_new_data_ISUP_cc')
    
    return rsz_inps, outs
def st_to_inp_out(st):
    w, h = 64, 3
    Topt = np.array([1,2])
    inps, outs, demos = getFurmanReg(st, Topt)
    outs = outs == 2
    ims, demos = compileImsDemos(inps, demos, w, h)
    rsz_inps = stack_obj_ar(ims)
    rsz_inps = [[im,d] for im, d in zip(rsz_inps, demos)]
    outs = list(outs)
    return rsz_inps, outs
    

def loadSt_orig():
    nm, demo = readCSVs('../clinical_data_renal_tumor.csv')
    nm = list(nm)
    
    drs = ['Penn_renal_tumor_segment_relabeled',
    'TCGA',
    'Xiangya Second Hospital',
    'Hunan People Hospital']
    drs = ['../../' + elt for elt in drs]
    imsLis, folLis, dicLis = [], [], []
    for dr in drs:
        dics, _, dirIms, dirFols = parseDir(dr)
        imsLis = imsLis + dirIms
        folLis = folLis + dirFols
        dicLis = dicLis + dics
    
    checkHasT1T2(imsLis, folLis)
    collect = collectInCSV(dicLis, folLis, nm, demo)
    '''
    tmp_szs = []
    for ims in imsLis:
        for im in ims:
            tmp_szs.append(im[1].shape)
    zs = [elt[2] for elt in tmp_szs]
    '''
    newStruct = []
    for row in collect:
        T1Im = -1
        T2Im = -1
        T2I = [i for i,elt in enumerate(row[1]) if elt[0]=='T2WI']
        T1C = [i for i,elt in enumerate(row[1]) if elt[0]=='T1C']
        if T1C:
            T1Im = row[1][T1C[0]][1:3]
            #T1Im, s = rsz3D(row[1][T1C[0]][1], 224, 7)
            #T1Im = getLargestSlice_Resize(row[1][T1C[0]][1], (224, 224))
        if T2I:
            T2Im = row[1][T2I[0]][1:3]
            # T2Im, s = rsz3D(row[1][T2I[0]][1], 224, 7)
            #T2Im = getLargestSlice_Resize(row[1][T2I[0]][1], (224, 224))
        newStruct.append([row[0], row[2], T1Im, T2Im])
    st = np.array(newStruct)
    return st
def loadStTest():
    nm1, demo1 = readCSVs('../clinical_data_renal_tumor.csv')
    nm2, demo2 = readCSVs('../../Penn_renal_tumor2/penn_renal_tumor2_formatted.csv')
    nm = list(nm1) + list(nm2)
    demo = np.vstack([demo1,demo2])

    nm = list(nm)

    
    
    drs = ['Penn_renal_tumor_segment_relabeled',
    'TCGA',
    'Xiangya Second Hospital',
    'Hunan People Hospital',
    'Penn_renal_tumor2']
    drs = ['../../' + elt for elt in drs]
    imsLis, folLis, dicLis = [], [], []
    for dr in drs:
        dics, _, dirIms, dirFols = parseDir(dr)
        imsLis = imsLis + dirIms
        folLis = folLis + dirFols
        dicLis = dicLis + dics
    
    checkHasT1T2(imsLis, folLis)
    collect = collectInCSV(dicLis, folLis, nm, demo)
    '''
    tmp_szs = []
    for ims in imsLis:
        for im in ims:
            tmp_szs.append(im[1].shape)
    zs = [elt[2] for elt in tmp_szs]
    '''
    newStruct = []
    for row in collect:
        T1Im = -1
        T2Im = -1
        T2I = [i for i,elt in enumerate(row[1]) if elt[0]=='T2WI']
        T1C = [i for i,elt in enumerate(row[1]) if elt[0]=='T1C']
        if T1C:
            T1Im = row[1][T1C[0]][1:3]
            #T1Im, s = rsz3D(row[1][T1C[0]][1], 224, 7)
            #T1Im = getLargestSlice_Resize(row[1][T1C[0]][1], (224, 224))
        if T2I:
            T2Im = row[1][T2I[0]][1:3]
            # T2Im, s = rsz3D(row[1][T2I[0]][1], 224, 7)
            #T2Im = getLargestSlice_Resize(row[1][T2I[0]][1], (224, 224))
        newStruct.append([row[0], row[2], T1Im, T2Im])
    st = np.array(newStruct)
    #save_obj(st, '../inp_new_penn')
    '''
w, h=224, 3
def getInps(spl = 0.3, Topt = np.array([1])):
			    Topt = np.array([1,2])
			    inps, outs, demos = getFurmanReg(st, Topt)
			    outs = outs == 1
			    ims, demos = compileImsDemos(inps, demos, w, h)
			    rsz_inps = stack_obj_ar(ims)
			    rsz_inps = [[im,d] for im, d in zip(rsz_inps, demos)]
                  outs = list(outs)
    '''

def loadSt2():
    nm1, demo1 = readCSVs('../RCC_histological_grade.csv')
    nm2, demo2 = readCSVs('../../Penn renal tumor2/penn_renal_tumor2_formatted.csv')
    nm = nm2 + nm2
    demo = np.vstack([demo1,demo2])

    nm = list(nm)
    
    drs = ['Penn_renal_tumor_segment_relabeled',
    'TCGA',
    'Xiangya Second Hospital',
    'Hunan People Hospital',
    'Penn renal tumor2']
    drs = ['../../' + elt for elt in drs]
    imsLis, folLis, dicLis = [], [], []
    for dr in drs:
        dics, _, dirIms, dirFols = parseDir(dr)
        imsLis = imsLis + dirIms
        folLis = folLis + dirFols
        dicLis = dicLis + dics
    
    checkHasT1T2(imsLis, folLis)
    collect = collectInCSV(dicLis, folLis, nm, demo)
    '''
    tmp_szs = []
    for ims in imsLis:
        for im in ims:
            tmp_szs.append(im[1].shape)
    zs = [elt[2] for elt in tmp_szs]
    '''
    newStruct = []
    for row in collect:
        T1Im = -1
        T2Im = -1
        T2I = [i for i,elt in enumerate(row[1]) if elt[0]=='T2WI']
        T1C = [i for i,elt in enumerate(row[1]) if elt[0]=='T1C']
        if T1C:
            T1Im = row[1][T1C[0]][1:3]
            #T1Im, s = rsz3D(row[1][T1C[0]][1], 224, 7)
            #T1Im = getLargestSlice_Resize(row[1][T1C[0]][1], (224, 224))
        if T2I:
            T2Im = row[1][T2I[0]][1:3]
            # T2Im, s = rsz3D(row[1][T2I[0]][1], 224, 7)
            #T2Im = getLargestSlice_Resize(row[1][T2I[0]][1], (224, 224))
        newStruct.append([row[0], row[2], T1Im, T2Im])
    st = np.array(newStruct)
    return st





def readCSVs_Fuhrman_ISUP(csv_path):
    data = pandas.read_csv(csv_path)
    nms = data.iloc[:,0].values
    
    inds = np.array([i for i,elt in enumerate(nms) if hasNumbers(elt)])
    
    
    nms = nms[inds]
    pathols = data.iloc[inds,1].values
    #hist_type(pathols)
    RCCs = data.iloc[inds,2].values
    #hist_type(RCCs)
    benigns = data.iloc[inds,3].values
    #hist_type(benigns)
    furmans = data.iloc[inds,4].values
    #hist_type(furmans)
    ages = data.iloc[inds,5].values
    #hist_type(ages)
    genders = data.iloc[inds,6].values
    #hist_type(genders)
    
    demos = np.vstack([RCCs, pathols, benigns, furmans, ages, genders])
    
    Fuhrman_vs_ISUP = data.iloc[inds,7].values
    return nms, np.transpose(demos), Fuhrman_vs_ISUP



def split(inps, outs, test_perc):
    '''
    test_perc = 0.3
    outs = np.hstack([np.zeros((10,)), np.ones((10,))])
    
    
    ar = (outs)
    '''
    ar = np.array(outs)
    uniq = np.unique(ar)
    tst_inds = []
    trn_inds = []

    for elt in uniq:
        inds_elt = np.where(ar==elt)[0]
        
        l = len(inds_elt)
        inds = np.arange(l)
        np.random.shuffle(inds)
        cutoff = int(np.ceil(l*test_perc))
        
        tst_inds.append(inds_elt[inds[:cutoff]])
        trn_inds.append(inds_elt[inds[cutoff:]])
    tst_inds = np.hstack(tst_inds)
    trn_inds = np.hstack(trn_inds)
    
    tstLs = list(ar[tst_inds])
    trnLs = list(ar[trn_inds])
    
    tstIms = [inps[i] for i in tst_inds]
    trnIms = [inps[i] for i in trn_inds]
    
    return trnLs, trnIms, tstLs, tstIms


def split_trn_tst_val_stochastic(inps, outs, trn_tst_val):
    #inps, outs, trn_tst_val = rsz_inps, outs, spl
    ar = np.array(outs)
    uniq = np.unique(ar)
    tst_inds = []
    trn_inds = []
    val_inds = []

    for elt in uniq:
        inds_elt = np.where(ar==elt)[0]
        
        l = len(inds_elt)
        inds = np.arange(l)
        np.random.shuffle(inds)

        cutoff_trn = int(np.ceil(l*trn_tst_val[0]))
        cutoff_tst = int(np.ceil(l*trn_tst_val[1])) + cutoff_trn
        #cutoff_val = int(np.ceil(l*trn_tst_val[2])) + cutoff_tst

        
        trn_inds.append(inds_elt[inds[:cutoff_trn]])
        tst_inds.append(inds_elt[inds[cutoff_trn:cutoff_tst]])
        val_inds.append(inds_elt[inds[cutoff_tst:]])
    tst_inds = np.hstack(tst_inds)
    trn_inds = np.hstack(trn_inds)
    val_inds = np.hstack(val_inds)
    
    tstLs = list(ar[tst_inds])
    trnLs = list(ar[trn_inds])
    valLs = list(ar[val_inds])
    
    tstIms = [inps[i] for i in tst_inds]
    trnIms = [inps[i] for i in trn_inds]
    valIms = [inps[i] for i in val_inds]
    
    return trnLs, trnIms, tstLs, tstIms, valLs, valIms

def split_trn_tst_val(inps, outs, trn_tst_val):


    inds = np.array(ld_obj('split_indices'))

    trn_inds = inds[:211]
    val_inds = inds[211:211+58]
    tst_inds = inds[211+58:]

    ar = np.array(outs)

    l = len(inds)


    tstLs = list(ar[tst_inds])
    trnLs = list(ar[trn_inds])
    valLs = list(ar[val_inds])
    
    tstIms = [inps[i] for i in tst_inds]
    trnIms = [inps[i] for i in trn_inds]
    valIms = [inps[i] for i in val_inds]
    
    return trnLs, trnIms, tstLs, tstIms, valLs, valIms

def split_trn_tst_val_rcc(inps, outs, trn_tst_val):


    inds = np.array(ld_obj('split_indices_rcc'))

    trn_inds = inds[:208]
    val_inds = inds[208:]

    ar = np.array(outs)

    l = len(inds)


    trnLs = list(ar[trn_inds])
    valLs = list(ar[val_inds])
    
    trnIms = [inps[i] for i in trn_inds]
    valIms = [inps[i] for i in val_inds]
    
    return trnLs, trnIms, valLs, valIms



def getLargestSlice_Resize(im, sz):
    sl_szs = np.sum(np.sum(im>0, axis = 0),axis=0)
    #rszIm = resize(im[:,:,np.argmax(sl_szs)], sz)
    #np.repeat(rszIm[:, :, np.newaxis], 3, axis=2)
    return resize(im[:,:,np.argmax(sl_szs)], sz)
def getLargestNSlice_Resize(im, sz, n):
    sl_szs = np.sum(np.sum(im>0, axis = 0),axis=0)
    topN = np.argsort(sl_szs)[-(n+1):]
    return resize(im[:,:,topN], sz)
def get_50_75_100_Slice_Resize(im, sz, n):
    sl_szs = np.sum(np.sum(im>0, axis = 0),axis=0)
    sz_50_75_100 = [
     np.argsort(sl_szs)[-1],
    np.argsort(sl_szs)[int(len(sl_szs)/2)],
    np.argsort(sl_szs)[int(len(sl_szs)*3/4)]
    ]
    return resize(im[:,:,sz_50_75_100], sz)

'''
#datawise
def preProcess(ls):
    inds = []
    valid_ims = []
    for i,elt in enumerate(ls):
        if np.all(elt != -1):
            
            valid_ims.append(elt)
            inds.append(i)
    ar = np.stack(valid_ims, axis = 0)
    processed = (ar-np.min(ar))/(np.max(ar-np.min(ar))/255)-114.7993278503418
    print(np.min(processed), np.mean(processed), np.max(processed))
    return [processed[inds.index(i)] if i in inds else -1 for i in range(len(ls))]
'''
def resRng(ar):
    return (ar-np.min(ar))/(np.max(ar-np.min(ar))/255)
'''     
def preProcess(ls):
    return [resRng(elt) if np.all(elt != -1) else -1 for elt in ls]
'''
def preProcess(ls):
    ims = []
    mn = np.inf
    for elt in ls:
        if np.all(elt != -1):
            mnCent = elt - np.mean(elt)
            unitVar = mnCent/np.std(mnCent)
            ims.append(unitVar)
    out = []
    i = 0
    for elt in ls:
        if np.all(elt != -1):
            out.append(ims[i]) 
            i = i + 1
        else:
            out.append(-1)
    return out, mn

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False


def hist_type(ar):
    for elt in np.unique(ar):
        print(str((elt))+':'+str(np.sum(ar==elt)))
    out = np.zeros((len(np.unique(ar)),))
    for i,elt in enumerate(np.unique(ar)):
        print(str((elt))+':'+str(np.mean(ar==elt)))
        out[i] = np.mean(ar==elt)
    return out

def getMalign(st, Topt):
    #get T1
    # isolate malign
    Tin = Topt + 1
    has_label_and_inp = np.vstack([elt for elt in st if RepresentsInt(elt[1][1]) and np.all(elt[Tin]!=-1)])
    outs = np.array([int(elt[1]) for elt in has_label_and_inp[:,1]]).astype(bool)
    inps = has_label_and_inp[:,Tin]
    demos = [elt[4:] for elt in has_label_and_inp[:, 1]]
    return inps, outs, demos
def getFurman(st, Topt):
    Tin = Topt + 1
    has_label_and_inp = np.vstack([elt for elt in st if RepresentsInt(elt[1][3]) and np.all(elt[Tin]!=-1)])
    outs = np.array([int(elt[3]) for elt in has_label_and_inp[:,1]])>2
    inps = has_label_and_inp[:,Tin]
    demos = [elt[4:] for elt in has_label_and_inp[:, 1]]
    return inps, outs, demos
def repIntEq1(n):
    if RepresentsInt(n):
        return int(n) == 1
    else:
        return False
def getFurmanRCC(st, Topt):
    Tin = Topt + 1
    has_label_and_inp = np.vstack([elt for elt in st if RepresentsInt(elt[1][3]) and np.all(elt[Tin]!=-1)])
    has_label_and_inp = np.vstack([elt for elt in has_label_and_inp if repIntEq1(elt[1][0])])
    outs = np.array([int(elt[3]) for elt in has_label_and_inp[:,1]])>2
    inps = has_label_and_inp[:,Tin]
    demos = [elt[4:] for elt in has_label_and_inp[:, 1]]
    return inps, outs, demos
def getFurmanReg(st, Topt):
    Tin = Topt + 1
    has_label_and_inp = np.vstack([elt for elt in st if RepresentsInt(elt[1][3]) and np.all(elt[Tin]!=-1)])
    outs = np.array([int(elt[3]) for elt in has_label_and_inp[:,1]])
    inps = has_label_and_inp[:,Tin]
    demos = [elt[4:] for elt in has_label_and_inp[:, 1]]
    return np.vstack(inps), outs, demos
def getHist(st, Topt):
    outs = []
    inps = []
    demos = []
    Tin = Topt + 1
    for elt in st:
        boolIm = np.all(elt[Tin]!=-1)
        benign = elt[1][1] == 0
        #if not benign:
        if RepresentsFloat(elt[1][0]):
            clear = float(elt[1][0]) == 1
        else:
            clear = '1' in elt[1][0]
            print(elt[1])
        not_clear = RepresentsInt(elt[1][0]) and '1' not in elt[1][0]
        
        if benign + clear + not_clear != 1:
            print(elt[0])
        elif boolIm:
            if benign:
                out = 0
            elif clear:
                out = 1
            elif not_clear:
                out = 2
            outs.append(out)
            inps.append(elt[Tin])
            demos.append(elt[1][4:])
    return np.vstack(inps), outs, demos
def getClearPapillary(st, Topt):
    outs = []
    inps = []
    demos = []
    Tin = Topt + 1
    for elt in st:
        boolIm = np.all(elt[Tin]!=-1)
        #if not benign:
        if RepresentsFloat(elt[1][0]):
            clear = float(elt[1][0]) == 1
        else:
            clear = '1' in elt[1][0]
            print(elt[1])
        pappilary = elt[1][0]=='2'
        
        if pappilary + clear > 1:
            print('fault')
        elif boolIm and (clear or pappilary):
            if clear:
                out = 0
            elif pappilary:
                out = 1
            outs.append(out)
            inps.append(elt[Tin])
            demos.append(elt[1][4:])
    np.sum(outs == 1)
    np.sum(outs == 0)
    return np.vstack(inps), outs, demos

def getRCCnotRCC(st, Topt):
    outs = []
    inps = []
    demos = []
    Tin = Topt + 1
    for elt in st:
        if RepresentsFloat(elt[1][0]):
            if np.isnan(float(elt[1][0])):
                continue

        boolIm = np.all(elt[Tin]!=-1)
        if RepresentsFloat(elt[1][0]):
            clear = float(elt[1][0]) == 1
        else:
            clear = '1' in elt[1][0]
            print(elt[1])
        not_clear = RepresentsInt(elt[1][0]) and '1' not in str(elt[1][0])
        
        if clear + not_clear > 1:
            print(elt[0])
        elif boolIm and (clear or not_clear):
            if clear:
                out = 0
            elif not_clear:
                out = 1
            outs.append(out)
            inps.append(elt[Tin])
            demos.append(elt[1][4:])
            #print(elt[1][0], out)
    return np.vstack(inps), outs, demos
def getClassWeights(ls):
    z_sum = float(np.sum(ls))/len(ls)
    classes = np.array([z_sum, 1-z_sum])
    classes = classes/min(classes)
    class_weight = {0: classes[1],
                1: classes[0]}
    return class_weight





# Import stuff
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    #image, alpha, sigma, alpha_affine = im, 1, 1, 1
    #im.shape[1]*0.25, im.shape[1] * 0.08, im.shape[1] * 0.08
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    #dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='constant').reshape(shape)


def single_im_batch_generator(batch_size, tstIms, lbls, nb_classes, w, hi):
    batch_inds_tst = np.arange(len(tstIms))
    np.random.shuffle(batch_inds_tst)
    while True:
        to_cut = min(batch_size, len(batch_inds_tst))
        use = batch_inds_tst[:to_cut]
        batch_inds_tst = batch_inds_tst[to_cut:]
        if len(batch_inds_tst)==0:
            batch_inds_tst = np.arange(len(tstIms))
            np.random.shuffle(batch_inds_tst)
        outIms = []
        outLs = []
        for i in use:
            rands = np.random.rand(4)
            im = tstIms[i]            
            '''
            plt.imshow(im)
            plotSlices(im)
            im2 = elastic_transform(im, 5, 2, 2)
            plotSlices(im2)
            '''
            # rotate, first rands
            d = rands[0]*360
            im = my_rot(im, d)
            
            # flip
            if rands[1]>0.5:
                im = np.fliplr(im)
            if rands[2]>0.5:
                im = np.flipud(im)
            
            
                
            # Gaussian noise
            a = im > np.mean(im)
            st_dev = np.std(im[a])
            im[a] = im[a] + (np.random.rand(np.sum(a))-0.5)*st_dev/4
            a
            
            im, _ = rsz3D(im, w, hi)
            ######################
            # elastic deformation
            '''
            if rands[3]>0.1:
                im = elastic_transform(im,2,2,2)
            '''
            # rescale x,y
            # crop to 100,100, resize? p=0.7?
            ######################
            im = (im-np.mean(im))/np.std(im)
            #im = im/255.0
            outIms.append(im)
            outLs.append(lbls[i].astype(float))
        outIms = np.stack(outIms, axis=0)
        yield (outIms, np_utils.to_categorical(outLs, nb_classes))



from skimage.transform import resize
def compileImsDemos_isolate_quartiles(inps, demos, w, hi):
    s = inps.shape
    newIms = np.zeros(s, dtype=object)
    newDemos = np.zeros(s, dtype=object)
    for i in range(s[0]):
        for j in range(s[1]):
            dbl = inps[i,j]
            im_demo, im = register(dbl)

            sl_sz_order = np.argsort(np.sum(np.sum(im>0, axis = 0),axis=0))
            sl_sz_len = len(sl_sz_order)
            percs = np.round(np.linspace(0.5, 1, hi) * sl_sz_len) - 1

            quarts = sl_sz_order[percs.astype(int)]
            
            
            newIms[i,j] = resize(im[:,:,quarts], (w,w))

            newDemos[i,j] = im_demo
            
    all_demos = np.hstack([np.vstack(demos), np.vstack([np.hstack(elt) for elt in newDemos])]).astype(float)
    all_demos[np.isnan(all_demos)]=-1
    return newIms, all_demos

#ims, demos = compileImsDemos(inps, demos)
'''
h,w = 8,112
nb_classes = 2
model1 = ResnetBuilder.build_resnet_18((h,w,w),nb_classes)
model2 = ResnetBuilder.build_resnet_18((h,w,w),nb_classes)
nd = 10
'''

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
    
    dense_soft = Lambda(lambda x: K.tf.nn.softmax(x))(dense)
    
    
    
    
    full_model = Model(inputs = [model1.input, model2.input, demo_inp], outputs = [dense_soft])
    return full_model


def dbl_im_batch_generator(batch_size, tstIms, lbls, nb_classes, w, hi):
    batch_inds_tst = np.arange(len(tstIms))
    np.random.shuffle(batch_inds_tst)
    while True:
        to_cut = min(batch_size, len(batch_inds_tst))
        use = batch_inds_tst[:to_cut]
        batch_inds_tst = batch_inds_tst[to_cut:]
        if len(batch_inds_tst)==0:
            batch_inds_tst = np.arange(len(tstIms))
            np.random.shuffle(batch_inds_tst)
        outIms = [[],[]]
        outLs = []
        demoL = []
        for i in use:
            rands = np.random.rand(5)
            ims, demo = tstIms[i]
            demoL.append(demo)
            d = rands[0]*360
            for j in range(2):
                im = ims[j]
                
                # rotate, first rands
                im = my_rot(im, d)
                
                # flip
                if rands[1]>0.5:
                    im = np.fliplr(im)
                if rands[2]>0.5:
                    im = np.flipud(im)
                
                
                    
                # Gaussian noise
                a = im > np.mean(im)
                st_dev = np.std(im[a])
                im[a] = im[a] + (np.random.rand(np.sum(a))-0.5)*st_dev/4
                
                
                im, _ = rsz3D(im, w, hi)
                '''
                ######################
                # elastic deformation
                if rands[3]>0.5:
                    im = elastic_transform(im,2,2,2)#
                ######################
                '''
                #im = (im-np.mean(im))/np.std(im)
                im = im/255.0
                outIms[j].append(im)
            outLs.append(lbls[i].astype(float))
        outIms0 = np.stack(outIms[0], axis=0)
        outIms1 = np.stack(outIms[1], axis=0)
        demos = np.stack(demoL, axis=0)
        yield ([outIms0, outIms1, demos], np_utils.to_categorical(outLs, nb_classes))
def addEndingDropOutRegularization(model, dr, kr, ar):
    model.layers.pop()
    x = model.layers[-1].output
    x = Dropout(dr)(x)
    x = Dense(nb_classes, activation='softmax', name='predictions', kernel_regularizer=regularizers.l2(kr), activity_regularizer=regularizers.l1(ar))(x)
    model = Model(outputs=x, inputs=model.input)
    return model


def replace_intermediate_layer_in_keras(model, layer_id, new_layer):

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model

def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model



def changeFinalLayer(model, nb_classes, kr, ar, dr, reg = False, m = False):
    x = model.layers[-3].output
    x = Dropout(dr)(x)
    dense = Dense(nb_classes, name='predictions', kernel_regularizer=regularizers.l2(kr), activity_regularizer=regularizers.l1(ar))(x)
    if not reg:
        dense_soft = Lambda(lambda x: K.tf.nn.softmax(x))(dense)
    else:
        dense_soft = Lambda(lambda x: K.tf.nn.sigmoid(x))(dense)
        if m:
            dense_soft = Lambda(lambda x: x * 4.0)(dense_soft)
    return Model(outputs=dense_soft, inputs=model.input)

def changeFinalLayer_no_sig(model, nb_classes, kr, ar, dr, reg=False):
    x = model.layers[-3].output
    x = Dropout(dr)(x)
    dense = Dense(nb_classes, name='predictions', kernel_regularizer=regularizers.l2(kr), activity_regularizer=regularizers.l1(ar))(x)
    dense_soft = dense
    return Model(outputs=dense_soft, inputs=model.input)


def changeFinalLayerReg(model, nb_classes, kr, ar, dr):
    x = model.layers[-3].output
    x = Dropout(dr)(x)
    dense = Dense(nb_classes, name='predictions', kernel_regularizer=regularizers.l2(kr), activity_regularizer=regularizers.l1(ar))(x)
    dense_soft = Lambda(lambda x: K.tf.nn.tanh(x))(dense)
    return Model(outputs=dense_soft, inputs=model.input)
def changeFinalLayerKRN(model, nb_classes, kr, ar, dr):
    x = model.layers[-1].output
    x = Dropout(dr)(x)
    dense = Dense(nb_classes, name='predictions', kernel_regularizer=regularizers.l2(kr), activity_regularizer=regularizers.l2(ar))(x)
    dense_soft = Lambda(lambda x: K.tf.nn.softmax(x))(dense)
    return Model(outputs=dense_soft, inputs=model.input)
def stack_obj_ar(ims):
    seqs = []
    for j in range(ims.shape[1]):
        seqs.append(np.stack(ims[:,j], axis=0))
    return np.stack(seqs, axis = 4)



def LR(nb_inp, nb_classes, reg = False, m=False):
    model = Sequential() 
    model.add(Dense(nb_classes, input_dim=nb_inp)) 
    if not reg:
        model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
    else:
        model.add(Lambda(lambda x: K.tf.nn.sigmoid(x)))
        if m:
            model.add(Lambda(lambda x: 4*x))
    return model
# test
m = LR(10,2)





'''
x_train, y_train = trnIms[:,:,:,:,0], np_utils.to_categorical(trnLs, nb_classes)
x_val, y_val = tstIms[:,:,:,:,0], np_utils.to_categorical(tstLs, nb_classes)

chknm = '../h5/T1'+fl_nm[:-2]+'h5'
learning_rate = 0.01
SGD_mom = 0.9
SGD_dec = 1e-6

kr, ar, dr = 0.01, 0.01, 0.5
n = 32
seq_s = 'T1'
'''
def seq_run(x_train, y_train, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, model1):
    mn = np.mean(x_train)
    s = np.std(x_train)
    
    x_train = np.stack([(im-np.mean(im))/np.std(im) for im in x_train],axis=0)
    x_val = np.stack([(im-np.mean(im))/np.std(im) for im in x_val],axis=0)
    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,shear_range=10,zoom_range=0.05
        )
    
    optim = SGD(lr=learning_rate, decay=SGD_dec, momentum=SGD_mom, nesterov=True)
    
    callbacks = [
            ModelCheckpoint(chknm, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1),
        ]
    
    
    model1.summary()
    
    print('model loaded')
    
    weights = 1/hist_type(np.array(tstLs+trnLs))
    weights = weights/np.min(weights)
    model1.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print('model compiled')
    
    
    stepsTrn = int(np.ceil(float(len(x_train))/n))
    
    print('train: '+str(np.mean(np.array(trnLs))))
    print('test: '+str(np.mean(np.array(tstLs))))
    
    hist_type(np.array(trnLs))
    hist_type(np.array(tstLs))
    '''
    for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
        for elt in X_batch:
            plotSlices(elt)
        break
    '''
    datagen.fit(x_train)
    
    
    history = model1.fit_generator(
            generator=datagen.flow(x_train, y_train, batch_size=n),
            epochs=10000,
            steps_per_epoch=stepsTrn,
            validation_data=(x_val, y_val),
            verbose=2,
            callbacks=callbacks)
    model1.save_weights('../h5/final_mdl_weights_' + seq_s + fl_nm[:-4])
    model1 = load_model(chknm, custom_objects={'tf':tf,})
    
    evl_tr = model1.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_ts = model1.evaluate(x=x_val, y=y_val)
    print('test: '+str(evl_ts))
    
    pred1_tr = model1.predict(x_train)
    pred1_ts = model1.predict(x_val)
    
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val, pred1_tr, pred1_ts]
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val, pred1_tr, pred1_ts], '../h5/' + seq_s + fl_nm[:-4])
    return pred1_tr, pred1_ts, collect
def seq_run_feat_cent(x_train, y_train, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, model, weighted=False, sv_root = ''):
    '''
    mn = np.mean(x_train)
    s = np.std(x_train)
    
    x_train = (x_train - mn)/s
    x_val = (x_val - mn)/s
    '''
    mn,s = -1,-1

    x_train = np.stack([(im-np.mean(im))/np.std(im) for im in x_train],axis=0)
    x_val = np.stack([(im-np.mean(im))/np.std(im) for im in x_val],axis=0)


    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,shear_range=10,zoom_range=0.05
        )
    
    optim = SGD(lr=learning_rate, decay=SGD_dec, momentum=SGD_mom, nesterov=True)

    callbacks = [
            ModelCheckpoint(chknm, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1),
        ]
    
    
    model.summary()
    
    print('model loaded')
    
    weights = 1/hist_type(np.array(tstLs+trnLs))
    if not weighted:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=optim, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])
    print('model compiled')
    
    
    stepsTrn = int(np.ceil(float(len(x_train))/n))
    
    print('train: '+str(np.mean(np.array(trnLs))))
    print('test: '+str(np.mean(np.array(tstLs))))
    
    hist_type(np.array(trnLs))
    hist_type(np.array(tstLs))
    '''
    for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
        for elt in X_batch:
            plotSlices(elt)
        break
    '''
    datagen.fit(x_train)
    
    
    history = model.fit_generator(
            generator=datagen.flow(x_train, y_train, batch_size=n),
            epochs=10000,
            steps_per_epoch=stepsTrn,
            validation_data=(x_val, y_val),
            verbose=2,
            callbacks=callbacks)
    model.save_weights('../h5/'+sv_root+'final_mdl_weights_' + seq_s + fl_nm[:-4])
    model.load_weights(chknm)
    
    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_ts = model.evaluate(x=x_val, y=y_val)
    print('test: '+str(evl_ts))

    my_globs.toPrint = my_globs.toPrint + seq_s + 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts)
    
    pred1_tr = model.predict(x_train)
    pred1_ts = model.predict(x_val)
    
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val, pred1_tr, pred1_ts]
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val, pred1_tr, pred1_ts], '../h5/' +sv_root+ seq_s + fl_nm[:-4])
    return pred1_tr, pred1_ts, collect

def seq_run_feat_cent3(x_train, y_train, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, model, weighted=False, sv_root = ''):
    '''
    mn = np.mean(x_train)
    s = np.std(x_train)
    
    x_train = (x_train - mn)/s
    x_val = (x_val - mn)/s
    '''
    mn,s = -1,-1

    x_train = np.stack([(im-np.mean(im))/np.std(im) for im in x_train],axis=0)
    x_val = np.stack([(im-np.mean(im))/np.std(im) for im in x_val],axis=0)


    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,shear_range=10,zoom_range=0.05
        )
    
    optim = SGD(lr=learning_rate, decay=SGD_dec, momentum=SGD_mom, nesterov=True)

    callbacks = [
            ModelCheckpoint(chknm, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1),
        ]
    
    
    model.summary()
    
    print('model loaded')
    
    weights = 1/hist_type(np.array(tstLs+trnLs))
    if not weighted:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=optim, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])
    print('model compiled')
    
    
    stepsTrn = int(np.ceil(float(len(x_train))/n))
    
    print('train: '+str(np.mean(np.array(trnLs))))
    print('test: '+str(np.mean(np.array(tstLs))))
    
    hist_type(np.array(trnLs))
    hist_type(np.array(tstLs))
    '''
    for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
        for elt in X_batch:
            plotSlices(elt)
        break
    '''
    datagen.fit(x_train)
    
    
    history = model.fit_generator(
            generator=datagen.flow(x_train, y_train, batch_size=n),
            epochs=10000,
            steps_per_epoch=stepsTrn,
            validation_data=(x_val, y_val),
            verbose=2,
            callbacks=callbacks)
    model.save_weights('../h5/'+sv_root+'final_mdl_weights_' + seq_s + fl_nm[:-4])
    model.load_weights(chknm)
    
    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_ts = model.evaluate(x=x_val, y=y_val)
    print('test: '+str(evl_ts))

    my_globs.toPrint = my_globs.toPrint + seq_s + 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts)
    
    pred1_tr = model.predict(x_train)
    pred1_ts = model.predict(x_val)
    
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val, pred1_tr, pred1_ts]
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val, pred1_tr, pred1_ts], '../h5/' +sv_root+ seq_s + fl_nm[:-4])
    return pred1_tr, pred1_ts, collect
import fnmatch

def seq_run_feat_cent_trn_val(x_train, y_train, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, trnLs, valLs, model, weighted=False, sv_root = '', epoch_num=10000, loss_tp='categorical_crossentropy'):
    mn,s = -1,-1

    x_train = np.stack([(im-np.mean(im))/np.std(im) for im in x_train],axis=0)
    x_val = np.stack([(im-np.mean(im))/np.std(im) for im in x_val],axis=0)


    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,shear_range=10,zoom_range=0.05
        )
    
    optim = SGD(lr=learning_rate, decay=SGD_dec, momentum=SGD_mom, nesterov=True)

    #modelNm = chknm[:-2] + 
    chckptNm = '../h5/'+sv_root+'chckpt'+seq_s+'.hdf5'
    callbacks = [
            ModelCheckpoint(chckptNm, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1),
        ]
    
    weights = 1/hist_type(np.array(trnLs+valLs))
    weights = weights/np.sum(weights)
    print(weights)
    if not weighted:
        model.compile(optimizer=optim, loss=loss_tp, metrics=['accuracy'])
    else:
        model.compile(optimizer=optim, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])
    
    
    stepsTrn = int(np.ceil(float(len(x_train))/n))
    

    datagen.fit(x_train)
    
    
    history = model.fit_generator(
            generator=datagen.flow(x_train, y_train, batch_size=n),
            epochs=epoch_num,
            steps_per_epoch=stepsTrn,
            validation_data=(x_val, y_val),
            verbose=2,
            callbacks=callbacks)
    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    model.load_weights(chckptNm)
    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    my_globs.toPrint = my_globs.toPrint + seq_s +'\n'+ 'train: ' + str(evl_tr) + 'val: ' + str(evl_vl)
    
    pred1_tr = model.predict(x_train)
    pred1_vl = model.predict(x_val)
    
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val, pred1_tr, pred1_vl, evl_tr, evl_vl]
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val, pred1_tr, pred1_vl], '../h5/' +sv_root+ seq_s + fl_nm[:-4])

    nm = sv_root
    tp = seq_s

    print(nm, tp)

    os.makedirs('../out/'+nm+tp)
    try:
        for p, y, trnvaltst in zip([pred1_tr, pred1_vl],[y_train, y_val],['trn','val']):
            my_globs.toPrint = my_globs.toPrint + '\n' + trnvaltst + '\n'
            dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
            with open('../out/'+nm+tp+'/'+trnvaltst+'dict.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in dic_one.items():
                    print(key)
                    print(value)
                    my_globs.toPrint = my_globs.toPrint + str((key, value))
                    writer.writerow([key, value])
    except Exception as e:
        print('except')

    my_globs.toPrint = my_globs.toPrint + '\n'
    return pred1_tr, pred1_vl, collect


def seq_run_feat_cent_trn_tst_val(x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model, weighted=False, sv_root = '', epoch_num=10000, loss_tp='categorical_crossentropy'):
    '''x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model1
    mn = np.mean(x_train)
    s = np.std(x_train)
    
    x_train = (x_train - mn)/s
    x_val = (x_val - mn)/s
    '''
    mn,s = -1,-1

    x_train = np.stack([(im-np.mean(im))/np.std(im) for im in x_train],axis=0)
    x_val = np.stack([(im-np.mean(im))/np.std(im) for im in x_val],axis=0)
    x_tst = np.stack([(im-np.mean(im))/np.std(im) for im in x_tst],axis=0)


    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,shear_range=10,zoom_range=0.05
        )
    
    optim = SGD(lr=learning_rate, decay=SGD_dec, momentum=SGD_mom, nesterov=True)

    #modelNm = chknm[:-2] + 
    chckptNm = '../h5/'+sv_root+'chckpt'+seq_s+'.hdf5'
    callbacks = [
            ModelCheckpoint(chckptNm, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1),
        ]
    
    
    #model.summary()
    
    #print('model loaded')
    
    weights = 1/hist_type(np.array(tstLs+trnLs+valLs))
    weights = weights/np.sum(weights)
    print(weights)
    if not weighted:
        model.compile(optimizer=optim, loss=loss_tp, metrics=['accuracy'])
    else:
        model.compile(optimizer=optim, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])
    #print('model compiled')
    
    
    stepsTrn = int(np.ceil(float(len(x_train))/n))
    

    datagen.fit(x_train)
    
    
    history = model.fit_generator(
            generator=datagen.flow(x_train, y_train, batch_size=n),
            epochs=epoch_num,
            steps_per_epoch=stepsTrn,
            validation_data=(x_val, y_val),
            verbose=2,
            callbacks=callbacks)
    '''
    seq_fls=[f for f in os.listdir('../h5/'+sv_root) if fnmatch.fnmatch(f, 'chckpt'+seq_s+'*')]
    mn_ind = np.argmin([float(re.match(r"^.*-(.*)\.hdf5$", f).group(1)) for f in seq_fls])
    mn_fl = seq_fls[mn_ind]
    '''
    #model.save_weights('../h5/'+sv_root+'final_mdl_weights_' + seq_s + fl_nm[:-4])
    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    model.load_weights(chckptNm)
    #model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    
    
    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))
    my_globs.toPrint = my_globs.toPrint + seq_s +'\n'+ 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts) + 'val: ' + str(evl_vl)
    
    pred1_tr = model.predict(x_train)
    pred1_vl = model.predict(x_val)
    pred1_ts = model.predict(x_tst)

    '''

    print(pred1_tr, y_train)
    print(pred1_tr.shape)
    print(pred1_tr.flatten().shape)
    print(np.array(y_train).shape)

    print(np.vstack([pred1_tr.flatten(), y_train]))
    '''
    
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts, evl_tr, evl_vl, evl_ts]
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts], '../h5/' +sv_root+ seq_s + fl_nm[:-4])

    nm = sv_root
    tp = seq_s

    print(nm, tp)

    os.makedirs('../out/'+nm+tp)
    try:
        for p, y, trnvaltst in zip([pred1_tr, pred1_vl, pred1_ts],[y_train, y_val, y_tst],['trn','val','tst']):
            my_globs.toPrint = my_globs.toPrint + '\n' + trnvaltst + '\n'
            dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
            with open('../out/'+nm+tp+'/'+trnvaltst+'dict.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in dic_one.items():
                    print(key)
                    print(value)
                    my_globs.toPrint = my_globs.toPrint + str((key, value))
                    writer.writerow([key, value])
    except Exception as e:
        print('except')

    my_globs.toPrint = my_globs.toPrint + '\n'
    return pred1_tr, pred1_ts, pred1_vl, collect


def ims_run(x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model_gen, weighted=False, sv_root = '', epoch_num=10000):
    '''x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model1
    '''
    mn,s = -1,-1

    x_train = np.stack([(im-np.mean(im))/np.std(im) for im in x_train],axis=0)
    x_val = np.stack([(im-np.mean(im))/np.std(im) for im in x_val],axis=0)
    x_tst = np.stack([(im-np.mean(im))/np.std(im) for im in x_tst],axis=0)

    model = model_gen()


    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,shear_range=10,zoom_range=0.05
        )
    
    optim = SGD(lr=learning_rate, decay=SGD_dec, momentum=SGD_mom, nesterov=True)

    #modelNm = chknm[:-2] + 
    chckptNm = '../h5/'+sv_root+'chckpt'+seq_s+'.hdf5'
    callbacks = [
            ModelCheckpoint(chckptNm, monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=40, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=300, verbose=1),
        ]
    
    weights = 1/hist_type(np.array(tstLs+trnLs+valLs))
    weights = weights/np.sum(weights)
    print(weights)
    if not weighted:
        loss_used = 'categorical_crossentropy'
    else:
        loss_used = weighted_categorical_crossentropy(weights)
    model.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])
    #print('model compiled')
    
    
    stepsTrn = int(np.ceil(float(len(x_train))/n))

    datagen.fit(x_train)
    
    history = model.fit_generator(
            generator=datagen.flow(x_train, y_train, batch_size=n),
            epochs=epoch_num,
            steps_per_epoch=stepsTrn,
            validation_data=(x_val, y_val),
            verbose=2,
            callbacks=callbacks)
    


    model.load_weights(chckptNm)
    model.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])

    
    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    
    
    pred1_tr = model.predict(x_train)
    pred1_vl = model.predict(x_val)
    pred1_ts = model.predict(x_tst)

    model.load_weights(chckptNm)

    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    pred1_tr = model.predict(x_train)
    pred1_vl = model.predict(x_val)
    pred1_ts = model.predict(x_tst)

    my_globs.toPrint = my_globs.toPrint + seq_s +'\n'+ 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts) + 'val: ' + str(evl_vl)

    '''
    del model

    K.clear_session()

    model_tmp = model_gen()
    model_tmp.load_weights(chckptNm)
    model_tmp.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])

    
    evl_tr = model_tmp.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model_tmp.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model_tmp.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    my_globs.toPrint = my_globs.toPrint + seq_s +'\n'+ 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts) + 'val: ' + str(evl_vl)
    
    pred1_tr = model_tmp.predict(x_train)
    pred1_vl = model_tmp.predict(x_val)
    pred1_ts = model_tmp.predict(x_tst)

    model_tmp.load_weights(chckptNm)

    pred1_tr = model_tmp.predict(x_train)
    pred1_vl = model_tmp.predict(x_val)
    pred1_ts = model_tmp.predict(x_tst)
    '''
    
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts]
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts], '../h5/' +sv_root+ seq_s + fl_nm[:-4])

    nm = sv_root
    tp = seq_s

    print(nm, tp)

    os.makedirs('../out/'+nm+tp)
    for p, y, trnvaltst in zip([pred1_tr, pred1_vl, pred1_ts],[y_train, y_val, y_tst],['trn','val','tst']):
        my_globs.toPrint = my_globs.toPrint + '\n' + trnvaltst + '\n'
        dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
        with open('../out/'+nm+tp+'/'+trnvaltst+'dict.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dic_one.items():
                print(key)
                print(value)
                my_globs.toPrint = my_globs.toPrint + str((key, value))
                writer.writerow([key, value])
    my_globs.toPrint = my_globs.toPrint + '\n'
    return pred1_tr, pred1_ts, pred1_vl, collect

def runDemoLR(nb_inp, nb_classes, batch_size, nb_epoch, trnDs, trnLs, tstDs, tstLs, fl_nm, tp_str='demo', process = True, weighted=False, sv_root = ''):
    chknm = '../h5/'+sv_root+tp_str+fl_nm[:-2]+'h5'
    callbacks = [
            ModelCheckpoint(chknm, monitor='val_loss', save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3000, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=30000, verbose=1),
        ]
    
    trnDs = np.stack(trnDs, axis=0)
    tstDs = np.stack(tstDs, axis=0)
    
    x_train, y_train = trnDs, np_utils.to_categorical(trnLs, nb_classes)
    x_val, y_val = tstDs, np_utils.to_categorical(tstLs, nb_classes)
    
    if process:
        mn = np.mean(x_train, axis = 0)
        s = np.std(x_train, axis = 0)
        x_train = (x_train - mn)/s
        x_val = (x_val - mn)/s
    
    model = LR(nb_inp, nb_classes)


    optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    
    weights = 1/hist_type(np.array(tstLs+trnLs))
    if not weighted:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=optim, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,verbose=2, validation_data=(x_val, y_val), callbacks=callbacks) 
    
    #model.save_weights('../h5/'+sv_root+'final_mdl_weights_'+tp_str+'_LR'+fl_nm[:-4])
    
    model.load_weights(chknm)
    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    
    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_ts = model.evaluate(x=x_val, y=y_val)
    print('test: '+str(evl_ts))
    my_globs.toPrint = my_globs.toPrint + tp_str + 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts)
    

    predD_tr = model.predict(x_train)
    predD_ts = model.predict(x_val)

    mn,s = -1,-1

    collectD = [history.history, mn, s, x_train, y_train, x_val, y_val, predD_tr, predD_ts]
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val, predD_tr, predD_ts], '../h5/'+sv_root+tp_str+ fl_nm[:-4])
    
    return predD_tr, predD_ts, collectD

def runDemoLR_trn_tst_val(nb_inp, nb_classes, batch_size, nb_epoch, trnDs, trnLs, tstDs, tstLs, valDs, valLs, fl_nm, tp_str='demo', process = True, weighted=False, sv_root = '', eval_train_xy=False):
    #nb_inp=len(tstDs[0])
    chckptNm = '../h5/'+sv_root+'chckpt'+tp_str+'.hdf5'
    callbacks = [
            ModelCheckpoint(chckptNm, monitor='val_loss', save_best_only=True, verbose=0, save_weights_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=300, min_lr=1e-9, epsilon=0.00001, verbose=0, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=3000, verbose=0),
        ]
    
    trnDs = np.stack(trnDs, axis=0)
    tstDs = np.stack(tstDs, axis=0)
    valDs = np.stack(valDs, axis=0)
    
    x_train, y_train = trnDs, np_utils.to_categorical(trnLs, nb_classes)
    x_tst, y_tst = tstDs, np_utils.to_categorical(tstLs, nb_classes)
    x_val, y_val = valDs, np_utils.to_categorical(valLs, nb_classes)

    mn,s = -1,-1
    if process:
        mn = np.mean(x_train, axis = 0)
        s = np.std(x_train, axis = 0)
        x_train = (x_train - mn)/s
        x_val = (x_val - mn)/s
        x_tst = (x_tst - mn)/s
    
    model = LR(nb_inp, nb_classes)


    optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    
    weights = 1/hist_type(np.array(tstLs+trnLs+valLs))
    weights = weights/np.sum(weights)
    #print(weights)
    if not weighted:
        loss_used = 'categorical_crossentropy'
    else:
        loss_used = weighted_categorical_crossentropy(weights)
    model.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,verbose=0, validation_data=(x_val, y_val), callbacks=callbacks) 
    
    model_tmp = model
    model_tmp.load_weights(chckptNm)
    model_tmp.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])

    if eval_train_xy:
        x_train = np.stack(eval_train_xy[0], axis=0)
        y_train = np_utils.to_categorical(eval_train_xy[1], nb_classes)
        if process:
            x_train = (x_train - mn)/s

    evl_tr = model_tmp.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model_tmp.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model_tmp.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    my_globs.toPrint = my_globs.toPrint + tp_str +'\n'+ 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts) + 'val: ' + str(evl_vl)

    pred1_tr = model_tmp.predict(x_train)
    #print(pred1_tr, y_train)
    pred1_vl = model_tmp.predict(x_val)
    pred1_ts = model_tmp.predict(x_tst)
    #[history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts]
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts, evl_tr, evl_vl, evl_ts]
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts], '../h5/' +sv_root+ tp_str + fl_nm[:-4])

    nm = sv_root
    tp = tp_str

    #print(nm, tp)

    os.makedirs('../out/'+nm+tp)
    for p, y, trnvaltst in zip([pred1_tr, pred1_vl, pred1_ts],[y_train, y_val, y_tst],['trn','val','tst']):
        dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
        with open('../out/'+nm+tp+'/'+trnvaltst+'dict.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dic_one.items():
                print(key, value)
                my_globs.toPrint = my_globs.toPrint + str((key, value))
                writer.writerow([key, value])
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts, evl_tr, evl_vl, evl_ts, dic_one]
    
    my_globs.toPrint = my_globs.toPrint + '\n'

    return pred1_tr, pred1_ts, pred1_vl, collect

def runDemoLR_trn_tst_val_reg(nb_inp, nb_classes, batch_size, nb_epoch, trnDs, trnLs, tstDs, tstLs, valDs, valLs, fl_nm, tp_str='demo', process = True, weighted=False, sv_root = '', eval_train_xy=False, loss_tp='categorical_crossentropy', reg=False, m=False):
    #nb_inp=len(tstDs[0])
    chckptNm = '../h5/'+sv_root+'chckpt'+tp_str+'.hdf5'
    callbacks = [
            ModelCheckpoint(chckptNm, monitor='val_loss', save_best_only=True, verbose=0, save_weights_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=300, min_lr=1e-9, epsilon=0.00001, verbose=0, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=3000, verbose=0),
        ]
    
    trnDs = np.stack(trnDs, axis=0)
    tstDs = np.stack(tstDs, axis=0)
    valDs = np.stack(valDs, axis=0)
    
    x_train, y_train = trnDs, trnLs
    x_tst, y_tst = tstDs, tstLs
    x_val, y_val = valDs, valLs
    mn,s = -1,-1
    if process:
        mn = np.mean(x_train, axis = 0)
        s = np.std(x_train, axis = 0)
        x_train = (x_train - mn)/s
        x_val = (x_val - mn)/s
        x_tst = (x_tst - mn)/s
    
    model = LR(nb_inp, nb_classes, reg=reg, m=m)


    optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    
    weights = 1/hist_type(np.array(tstLs+trnLs+valLs))
    weights = weights/np.sum(weights)
    #print(weights)
    if not weighted:
        loss_used = loss_tp
    else:
        loss_used = weighted_categorical_crossentropy(weights)
    model.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,verbose=0, validation_data=(x_val, y_val), callbacks=callbacks) 
    
    model_tmp = model
    model_tmp.load_weights(chckptNm)
    model_tmp.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])

    if eval_train_xy:
        x_train = np.stack(eval_train_xy[0], axis=0)
        y_train = np_utils.to_categorical(eval_train_xy[1], nb_classes)
        if process:
            x_train = (x_train - mn)/s

    evl_tr = model_tmp.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model_tmp.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model_tmp.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    my_globs.toPrint = my_globs.toPrint + tp_str +'\n'+ 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts) + 'val: ' + str(evl_vl)

    pred1_tr = model_tmp.predict(x_train)
    #print(pred1_tr, y_train)
    pred1_vl = model_tmp.predict(x_val)
    pred1_ts = model_tmp.predict(x_tst)
    #[history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts]
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts, evl_tr, evl_vl, evl_ts]
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts], '../h5/' +sv_root+ tp_str + fl_nm[:-4])

    nm = sv_root
    tp = tp_str

    #print(nm, tp)

    os.makedirs('../out/'+nm+tp)
    '''
    for p, y, trnvaltst in zip([pred1_tr, pred1_vl, pred1_ts],[y_train, y_val, y_tst],['trn','val','tst']):
        dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
        with open('../out/'+nm+tp+'/'+trnvaltst+'dict.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dic_one.items():
                print(key)
                print(value)
                my_globs.toPrint = my_globs.toPrint + str((key, value))
                writer.writerow([key, value])
    my_globs.toPrint = my_globs.toPrint + '\n'
    '''

    return pred1_tr, pred1_ts, pred1_vl, collect

class CustomModelCheckpoint(Callback):

    def __init__(self, model, path):

        super().__init__()

        # This is the argument that will be modify by fit_generator
        # self.model = model
        self.path = path

        # We set the model (non multi gpu) under an other name
        self.model_for_saving = model
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):

        loss = logs['val_loss']

        if loss <= self.best:
            self.best = loss
            #save
            print("\nSaving model to : {}".format(self.path.format(epoch=epoch, val_loss=loss))+' loss: '+str(loss))
            self.model_for_saving.save_weights(self.path.format(epoch=epoch, val_loss=loss), overwrite=True)
        else:
            #not save
            print("did not improve from "+str(self.best))


        # Here we save the original one
        

def ims_run2(x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model_gen, weighted=False, sv_root = '', epoch_num=10000):
    '''x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model1
    '''
    mn,s = -1,-1

    x_train = np.stack([(im-np.mean(im))/np.std(im) for im in x_train],axis=0)
    x_val = np.stack([(im-np.mean(im))/np.std(im) for im in x_val],axis=0)
    x_tst = np.stack([(im-np.mean(im))/np.std(im) for im in x_tst],axis=0)

    model = model_gen()


    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,shear_range=10,zoom_range=0.05
        )
    
    optim = SGD(lr=learning_rate, decay=SGD_dec, momentum=SGD_mom, nesterov=True)

    

    '''
    chckptNm = '../h5/'+sv_root+'chckpt'+seq_s+ 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    
    callbacks = [
            ModelCheckpoint(chckptNm, monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=40, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=300, verbose=1),
        ]
    '''
    chckptNm = '../h5/'+sv_root+'chckpt'+seq_s+'.hdf5'
    callbacks = [
            CustomModelCheckpoint(model, chckptNm),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=40, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=300, verbose=1),
        ]
    
    weights = 1/hist_type(np.array(tstLs+trnLs+valLs))
    weights = weights/np.sum(weights)
    print(weights)
    if not weighted:
        loss_used = 'categorical_crossentropy'
    else:
        loss_used = weighted_categorical_crossentropy(weights)
    model.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])
    #print('model compiled')
    
    
    stepsTrn = int(np.ceil(float(len(x_train))/n))

    datagen.fit(x_train)
    
    history = model.fit_generator(
            generator=datagen.flow(x_train, y_train, batch_size=n),
            epochs=epoch_num,
            steps_per_epoch=stepsTrn,
            validation_data=(x_val, y_val),
            verbose=2,
            callbacks=callbacks)
    '''
    seq_fls=[f for f in os.listdir('../h5/'+sv_root) if fnmatch.fnmatch(f, 'chckpt'+seq_s+'*')]
    mn_ind = np.argmin([float(re.match(r"^.*-(.*)\.hdf5$", f).group(1)) for f in seq_fls])
    mn_fl = seq_fls[mn_ind]
    chckptNm = mn_fl


    seq_fls.remove(chckptNm)
    for elt in seq_fls:
        os.remove('../h5/' + sv_root + elt)
    '''

    #model.save_weights('../h5/'+sv_root+'final_mdl_weights_' + seq_s + fl_nm[:-4])
    


    model.load_weights(chckptNm)
    model.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])

    
    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    
    
    pred1_tr = model.predict(x_train)
    pred1_vl = model.predict(x_val)
    pred1_ts = model.predict(x_tst)

    model.load_weights(chckptNm)

    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    pred1_tr = model.predict(x_train)
    pred1_vl = model.predict(x_val)
    pred1_ts = model.predict(x_tst)


    my_globs.toPrint = my_globs.toPrint + seq_s +'\n'+ 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts) + 'val: ' + str(evl_vl)

    '''
    del model

    K.clear_session()

    model_tmp = model_gen()
    model_tmp.load_weights(chckptNm)
    model_tmp.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])

    
    evl_tr = model_tmp.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model_tmp.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model_tmp.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    my_globs.toPrint = my_globs.toPrint + seq_s +'\n'+ 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts) + 'val: ' + str(evl_vl)
    
    pred1_tr = model_tmp.predict(x_train)
    pred1_vl = model_tmp.predict(x_val)
    pred1_ts = model_tmp.predict(x_tst)

    model_tmp.load_weights(chckptNm)

    pred1_tr = model_tmp.predict(x_train)
    pred1_vl = model_tmp.predict(x_val)
    pred1_ts = model_tmp.predict(x_tst)
    '''
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts]
    
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts], '../h5/' +sv_root+ seq_s + fl_nm[:-4])

    nm = sv_root
    tp = seq_s

    print(nm, tp)

    os.makedirs('../out/'+nm+tp)
    for p, y, trnvaltst in zip([pred1_tr, pred1_vl, pred1_ts],[y_train, y_val, y_tst],['trn','val','tst']):
        my_globs.toPrint = my_globs.toPrint + '\n' + trnvaltst + '\n'
        dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
        with open('../out/'+nm+tp+'/'+trnvaltst+'dict.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dic_one.items():
                print(key)
                print(value)
                my_globs.toPrint = my_globs.toPrint + str((key, value))
                writer.writerow([key, value])
    my_globs.toPrint = my_globs.toPrint + '\n'

    return pred1_tr, pred1_ts, pred1_vl, collect
import operator
def ims_run3(x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model_gen, weighted=False, sv_root = '', epoch_num=10000, eval_train_xy=False):
    '''x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model1
    '''
    mn,s = -1,-1

    x_train = np.stack([(im-np.mean(im))/np.std(im) for im in x_train],axis=0)
    x_val = np.stack([(im-np.mean(im))/np.std(im) for im in x_val],axis=0)
    x_tst = np.stack([(im-np.mean(im))/np.std(im) for im in x_tst],axis=0)

    model = model_gen()


    
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,shear_range=10,zoom_range=0.05
        )
    
    optim = SGD(lr=learning_rate, decay=SGD_dec, momentum=SGD_mom, nesterov=True)

    

    
    chckptNm = '../h5/'+sv_root+'chckpt'+seq_s+ 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    
    callbacks = [
            ModelCheckpoint(chckptNm, monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1),
        ]
    
    weights = 1/hist_type(np.array(tstLs+trnLs+valLs))
    weights = weights/np.sum(weights)
    print(weights)
    if not weighted:
        loss_used = 'categorical_crossentropy'
    else:
        loss_used = weighted_categorical_crossentropy(weights)
    model.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])
    #print('model compiled')
    
    
    stepsTrn = int(np.ceil(float(len(x_train))/n))

    datagen.fit(x_train)
    
    history = model.fit_generator(
            generator=datagen.flow(x_train, y_train, batch_size=n),
            epochs=epoch_num,
            steps_per_epoch=stepsTrn,
            validation_data=(x_val, y_val),
            verbose=2,
            callbacks=callbacks)
    


    seq_fls=[f for f in os.listdir('../h5/'+sv_root) if fnmatch.fnmatch(f, 'chckpt'+seq_s+'*')]

    loss = []
    epoch = []
    inds=[]
    for i,f in enumerate(seq_fls):
        loss.append(float(re.match(r"^.*-(.*)\.hdf5$", f).group(1)))
        epoch.append(-float(re.match(r"^.*\.(.*)-.*\.hdf5$", f).group(1)))
        inds.append(i)

    numsMat = np.transpose(np.vstack([loss, epoch, inds]).astype(float))


    numsMat = np.array(sorted(numsMat, key=operator.itemgetter(0,1)))

    order = numsMat[:,2].astype(int)

    seqs = np.array(seq_fls)[order]

    losses_srt = np.array(numsMat)[:,0].astype(float)

    toDel = np.where(losses_srt != np.min(losses_srt))[0]
    toRmv = seqs[toDel]

    for elt in toRmv:
        os.remove('../h5/'+sv_root+elt)

    chckptNm = str(seqs[0])
    print(chckptNm)

    if eval_train_xy:
        x_train = np.stack([(im-np.mean(im))/np.std(im) for im in eval_train_xy[0]],axis=0)
        y_train = eval_train_xy[1]
    


    model.load_weights('../h5/' + sv_root + chckptNm)
    model.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])

    
    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    
    
    pred1_tr = model.predict(x_train)
    pred1_vl = model.predict(x_val)
    pred1_ts = model.predict(x_tst)

    model.load_weights('../h5/' + sv_root + chckptNm)

    evl_tr = model.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    pred1_tr = model.predict(x_train)
    pred1_vl = model.predict(x_val)
    pred1_ts = model.predict(x_tst)


    my_globs.toPrint = my_globs.toPrint + seq_s +'\n'+ 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts) + 'val: ' + str(evl_vl)

    '''
    del model

    K.clear_session()

    model_tmp = model_gen()
    model_tmp.load_weights(chckptNm)
    model_tmp.compile(optimizer=optim, loss=loss_used, metrics=['accuracy'])

    
    evl_tr = model_tmp.evaluate(x=x_train, y=y_train)
    print('train: '+str(evl_tr))
    
    evl_vl = model_tmp.evaluate(x=x_val, y=y_val)
    print('val: '+str(evl_vl))
    
    evl_ts = model_tmp.evaluate(x=x_tst, y=y_tst)
    print('test: '+str(evl_ts))

    my_globs.toPrint = my_globs.toPrint + seq_s +'\n'+ 'train: ' + str(evl_tr) + 'test: ' + str(evl_ts) + 'val: ' + str(evl_vl)
    
    pred1_tr = model_tmp.predict(x_train)
    pred1_vl = model_tmp.predict(x_val)
    pred1_ts = model_tmp.predict(x_tst)

    model_tmp.load_weights(chckptNm)

    pred1_tr = model_tmp.predict(x_train)
    pred1_vl = model_tmp.predict(x_val)
    pred1_ts = model_tmp.predict(x_tst)
    '''
    collect = [history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts]
    
    save_obj([history.history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts], '../h5/' +sv_root+ seq_s + fl_nm[:-4])

    nm = sv_root
    tp = seq_s

    print(nm, tp)

    os.makedirs('../out/'+nm+tp)
    for p, y, trnvaltst in zip([pred1_tr, pred1_vl, pred1_ts],[y_train, y_val, y_tst],['trn','val','tst']):
        my_globs.toPrint = my_globs.toPrint + '\n' + trnvaltst + '\n'
        dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
        with open('../out/'+nm+tp+'/'+trnvaltst+'dict.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dic_one.items():
                print(key)
                print(value)
                my_globs.toPrint = my_globs.toPrint + str((key, value))
                writer.writerow([key, value])
    my_globs.toPrint = my_globs.toPrint + '\n'

    return pred1_tr, pred1_ts, pred1_vl, collect

import keras
def shareResNet(h,w, model, dr, kr, ar):
    def slice(x):
        global z
        print(z)
        return x[:, :,:,z]
    global z
    inpsSinglesOut = []
    stackInp = Input(shape=(w,w,h))
    for z in range(stackInp.shape[3]):
        sInp = Lambda(slice)(stackInp)
        sInp_r = Reshape((int(stackInp.shape[1]), int(stackInp.shape[2]), 1))(sInp)
        out_z = model(sInp_r)
        inpsSinglesOut.append(out_z)
        #full_model = Model(inputs = [stackInp], outputs = [out_z])
    merged_vector = keras.layers.concatenate(inpsSinglesOut, axis=-1)
    
    #full_model = Model(inputs = [stackInp], outputs = [merged_vector])
    
    dense_3 = Dense(24,)(merged_vector)
    x = Dropout(dr)(dense_3)
    dense = Dense(nb_classes, name='predictions', kernel_regularizer=regularizers.l2(kr), activity_regularizer=regularizers.l1(ar))(x)
    dense_soft = Lambda(lambda x: K.tf.nn.softmax(x))(dense)
    
    full_model = Model(inputs = [stackInp], outputs = [dense_soft])
    full_model.summary()
    return full_model
dr = '../grade'