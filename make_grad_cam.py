import os
from tsne import tsne
from matplotlib import pyplot as plt
from skimage.transform import resize
from multiprocessing import Pool


import my_globs
from vis.visualization import visualize_cam, visualize_saliency, overlay
from vis.utils.utils import load_img, normalize, find_layer_idx

from keras.models import Model
my_globs.initGLobals()
import cv2
from grad_cam import *

from utils import LR, ld_obj, runDemoLR_trn_tst_val_reg, getFurmanReg, save_obj, runDemoLR_trn_tst_val,seq_run_feat_cent_trn_tst_val, split_trn_tst_val,perfMeasures, seq_run_feat_cent, stack_obj_ar, runDemoLR, getMdlInps, save_obj, parseDir, rsz, weighted_categorical_crossentropy, my_rot, rsz3D, loadSt, split, plotSlices, getMalign, getClassWeights, hist_type, seq_run, register, getFurman, elastic_transform, single_im_batch_generator, compileImsDemos, LR, loadSt_orig, changeFinalLayer
import numpy as np

import os
import tensorflow as tf
from time import gmtime, strftime
import sys
sys.path.insert(0, 'keras-resnet-master')
from resnet_stoch_depth import ResnetBuilder

from keras import backend as K
from keras.utils import np_utils

from keras.optimizers import SGD
import scipy


def rsz(im):
    return resize(im, (224,224))
locn1 = '../Bag_Prob/chckptT1.h5'
locn2 = '../Bag_Prob/chckptT2.h5'

w = 64
h = 3
nb_classes = 2
model = ResnetBuilder.build_resnet_18((h,w,w),nb_classes)
kr, ar, dr = 0.1, 0.01, 0.5
optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = changeFinalLayer(model, nb_classes, kr, ar, dr)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(locn1)

model.summary()

def imadj(im):
    tmp = im - np.min(im)
    return tmp/np.max(tmp)*255
    


def getInps(spl = 0.3, Topt = np.array([1])):
    [trnLs, trnIms, tstLs, tstIms, valLs, valIms] = ld_obj('../data/inps')
    return trnLs, trnIms, tstLs, tstIms, valLs, valIms


trnLs, trnIms, tstLs, tstIms, valLs, valIms = getInps()


trnDs = [elt[1] for elt in trnIms]
trnIms = [elt[0] for elt in trnIms]

tstDs = [elt[1] for elt in tstIms]
tstIms = [elt[0] for elt in tstIms]

valDs = [elt[1] for elt in valIms]
valIms = [elt[0] for elt in valIms]

trnIms = np.stack(trnIms, axis = 0)
tstIms = np.stack(tstIms, axis = 0)
valIms = np.stack(valIms, axis = 0)

x_train, y_train = trnIms[:,:,:,:,0], np_utils.to_categorical(trnLs, nb_classes)
x_val, y_val = valIms[:,:,:,:,0], np_utils.to_categorical(valLs, nb_classes)
x_tst, y_tst = tstIms[:,:,:,:,0], np_utils.to_categorical(tstLs, nb_classes)

x_train = np.stack([(im-np.mean(im))/np.std(im) for im in x_train],axis=0)
x_val = np.stack([(im-np.mean(im))/np.std(im) for im in x_val],axis=0)
x_tst = np.stack([(im-np.mean(im))/np.std(im) for im in x_tst],axis=0)

evl_tr = model.evaluate(x=x_train, y=y_train)
print('train: '+str(evl_tr))

evl_vl = model.evaluate(x=x_val, y=y_val)
print('val: '+str(evl_vl))

evl_ts = model.evaluate(x=x_tst, y=y_tst)
print('test: '+str(evl_ts))


ls=model.layers
nms=[l.name for l in ls]
conv_ls=[i for i,elt in enumerate(nms) if 'conv' in elt]
nms.index('conv2d_19')
'''
for i in range(len(ls)):
    try:
        grad = visualize_cam(model, i, None, preprocessed_input, backprop_modifier='relu')
        plt.imshow(grad)
        cv2.imwrite(fol+str(i)+"new_grad"+center_str+".jpg", grad)
    except Exception as e:
        print(i)
        print(e)
        
for vers in [4,5,10,12,28,31,51]:
    grad = visualize_cam(model, i, None, preprocessed_input, backprop_modifier='relu')
    plt.imshow(grad)
    cv2.imwrite(fol+str(i)+'_'+str(vers)+"new_grad"+center_str+".jpg", grad)
        
        
        
grad = visualize_cam(model, len(ls)-1, None, preprocessed_input, backprop_modifier='relu',penultimate_layer_idx=64)
plt.imshow(grad)
cv2.imwrite(fol+str(i)+"final"+center_str+".jpg", grad)
    
try:
    pass
except Exception as e:
    raise e
'''



def my_grad_cam(ims, ts, model, Ls):
    #ims, ts, model, Ls=ims, 'T2',model, Ls
    plt.figure()

    for i,im in enumerate(ims):
        L = Ls[i]

        fol = "../18_opt/" + ts + '/'

        center_str = str(i)
        
        rpt_im = np.repeat(im[:,:,0][:, :, np.newaxis], 3, axis=2)
        
        preprocessed_input = np.expand_dims(im, axis=0)
        predictions = model.predict(preprocessed_input)
        predicted_class = np.argmax(predictions)

        if L == predicted_class:
            fol = fol+'Cor/'+str(L)+'_'+str(predicted_class)+'/'
        else:
            fol = fol+'InC/'+str(L)+'_'+str(predicted_class)+'/'
        if not os.path.exists(fol):
            os.makedirs(fol)
            
            
        for elt,tpp in zip([im, rpt_im],['norm','rpt']):
            z=74 
            rot=''
            for pn in [8,21]:
                preprocessed_input = np.expand_dims(elt, axis=0)
                grad = visualize_cam(model, z, None, preprocessed_input,penultimate_layer_idx=pn)
                vis_img = imadj(elt)[:,:,0]
                '''
                for cl in conv_ls:
                    grad = visualize_cam(model, 76, None, preprocessed_input,penultimate_layer_idx=cl)
                    cv2.imwrite(fol+'LOOK'+tpp+'heatmap_lay_'+str(cl)+"_"+str(z)+"_"+center_str+".jpg", grad)
                #grad = visualize_cam(model, 76, None, preprocessed_input,penultimate_layer_idx=16)
                '''
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(grad)
                ax[0].axis('off')
                ax[1].imshow(vis_img)
                ax[1].axis('off')
                
                fig.savefig(fol+tpp+'double_lay_'+str(pn)+"_"+center_str+".jpg")
                plt.close('all')



ims = np.vstack([x_train, x_val, x_tst])

Ls = trnLs+valLs+tstLs
#my_grad_cam(ims,'T1',model, Ls)

model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(locn2)

x_train, y_train = trnIms[:,:,:,:,1], np_utils.to_categorical(trnLs, nb_classes)
x_val, y_val = valIms[:,:,:,:,1], np_utils.to_categorical(valLs, nb_classes)
x_tst, y_tst = tstIms[:,:,:,:,1], np_utils.to_categorical(tstLs, nb_classes)

x_train = np.stack([(im-np.mean(im))/np.std(im) for im in x_train],axis=0)
x_val = np.stack([(im-np.mean(im))/np.std(im) for im in x_val],axis=0)
x_tst = np.stack([(im-np.mean(im))/np.std(im) for im in x_tst],axis=0)

evl_tr = model.evaluate(x=x_train, y=y_train)
print('train: '+str(evl_tr))

evl_vl = model.evaluate(x=x_val, y=y_val)
print('val: '+str(evl_vl))

evl_ts = model.evaluate(x=x_tst, y=y_tst)
print('test: '+str(evl_ts))

ims = np.vstack([x_train, x_val, x_tst])

Ls = trnLs+valLs+tstLs

my_grad_cam(ims,'T2',model, Ls)