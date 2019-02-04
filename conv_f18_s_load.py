#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:22:22 2018

@author: priscillachang
"""
import os
from tsne import tsne
from matplotlib import pyplot as plt


import my_globs

from keras.models import Model
my_globs.initGLobals()
import cv2
from grad_cam import *

from utils import LR, ld_obj, runDemoLR_trn_tst_val_reg, getFurmanReg, save_obj, runDemoLR_trn_tst_val,seq_run_feat_cent_trn_tst_val, split_trn_tst_val,perfMeasures, seq_run_feat_cent, stack_obj_ar, runDemoLR, getMdlInps, save_obj, parseDir, rsz, weighted_categorical_crossentropy, my_rot, rsz3D, loadSt, split, plotSlices, getMalign, getClassWeights, hist_type, seq_run, register, getFurman, elastic_transform, single_im_batch_generator, compileImsDemos, LR, loadSt_orig, changeFinalLayer
import numpy as np

import os
from time import gmtime, strftime
import sys
sys.path.insert(0, 'keras-resnet-master')
from resnet_stoch_depth import ResnetBuilder

from keras import backend as K
from keras.utils import np_utils

from keras.optimizers import SGD



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




def my_grad_cam(ims, lay, ts, model, Ls):
    #ims, lay, ts, model, Ls=ims, lay, 'T2',model, Ls
    plt.figure()

    for i,elt in enumerate(ims):
        L = Ls[i]

        fol = "../18_s/" + ts + '/' + lay + '/'

        center_str = str(i)

        preprocessed_input = np.expand_dims(elt, axis=0)
        predictions = model.predict(preprocessed_input)
        predicted_class = np.argmax(predictions)
        print(L,predicted_class)


        if L == predicted_class:
            fol = fol+'Cor/'+str(L)+'_'+str(predicted_class)+'/'
        else:
            fol = fol+'InC/'+str(L)+'_'+str(predicted_class)+'/'
        if not os.path.exists(fol):
            os.makedirs(fol)


        
        cv2.imwrite(fol+"im"+center_str+".jpg", 255*np.hstack([elt[:,:,0],elt[:,:,1],elt[:,:,2]]))

        image = elt - np.min(elt)
        image = np.minimum(image*255.0, 255)
        #print(np.max(image), np.min(image))
        image = np.float32(image)
        cv2.imwrite(fol+"image"+center_str+".jpg", np.hstack([image[:,:,0],image[:,:,1],image[:,:,2]]))







        #print(L, predicted_class)
        #print(L.shape, predicted_class.shape)

        #print(predicted_class)
        cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, lay)
        
        ####
        print(heatmap)
        print(heatmap.shape)
        print(np.mean(heatmap))
        cv2.imwrite(fol+"heatmap__"+center_str+".jpg", heatmap)
        print(fol+"heatmap__"+center_str+".jpg")
        
        ####
        '''
        print(cam.shape, heatmap.shape)
        print(cam)
        print(heatmap)
        '''
        cv2.imwrite(fol+"gradcam"+center_str+".jpg", cam)
        tmp = np.hstack([cam[:,:,0],cam[:,:,1],cam[:,:,2]])
        cv2.imwrite(fol+"gradcamSep"+center_str+".jpg", tmp)

        

        register_gradient()
        guided_model = modify_backprop(model, 'GuidedBackProp')
        saliency_fn = compile_saliency_function(guided_model, activation_layer=lay)
        saliency = saliency_fn([preprocessed_input, 0])
        '''
        print(len(saliency))
        print(saliency[0].shape)
        '''
        gradcam = saliency[0] * heatmap[..., np.newaxis]

        ####
        print(gradcam)
        print(deprocess_image(gradcam))
        sys.exit()
        ####



        cv2.imwrite(fol+"guided_gradcam"+center_str+".jpg", deprocess_image(gradcam))
        cv2.imwrite(fol+"cmpImGrad"+center_str+".jpg", np.hstack([image/np.max(image)*255, deprocess_image(gradcam)]))
        cv2.imwrite(fol+"cmpImGrad__"+center_str+".jpg", np.hstack([image/np.max(image)*255, imadj(gradcam)[0]]))
        
        dp_g = deprocess_image(gradcam)
        cv2.imwrite(fol+"cmpGradPcs"+center_str+".jpg", np.hstack([dp_g[:,:,0],dp_g[:,:,1],dp_g[:,:,2]]))
        tmp=np.vstack([np.hstack([imadj(dp_g[:,:,0]),imadj(dp_g[:,:,1]),imadj(dp_g[:,:,2])]), np.hstack([imadj(image[:,:,0]),imadj(image[:,:,1]),imadj(image[:,:,2])])])
        cv2.imwrite(fol+"cmpGradImPcs"+center_str+".jpg", tmp)

        plt.subplot(2, 1, 1)
        plt.imshow(np.hstack([dp_g[:,:,0],dp_g[:,:,1],dp_g[:,:,2]]))
        plt.subplot(2, 1, 2)
        image = elt - np.min(elt)
        plt.imshow(np.hstack([image[:,:,0],image[:,:,1],image[:,:,2]]))

        plt.savefig(fol+"Grad_Im"+center_str+".jpg")
        


lay = 'conv2d_19'
lay = 'conv2d_15'

ims = np.vstack([x_train, x_val, x_tst])

Ls = trnLs+valLs+tstLs
'''
my_grad_cam(ims, lay, 'T1',model, Ls)
my_grad_cam(ims, 'conv2d_19', 'T1',model, Ls)
'''


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

lay = 'conv2d_15'
my_grad_cam(ims, lay, 'T2',model, Ls)
my_grad_cam(ims, 'conv2d_19', 'T2',model, Ls)

sys.exit()



def cd(outFl):
    out = os.getcwd()
    os.chdir(outFl)
    return out

def with_substr(ls, sub_str):
    return [elt for elt in ls if sub_str in elt]

dataDir = '../h5/T223_06_34conv_f18.h5'
try:
    orig_dir = cd(dataDir)
except Exception as e:
    orig_dir = cd('/Users/marcello/Desktop/22_09_16conv_f18_s_reg_trn_tst_val_rpt2_15_01')
##just nav to directory

data_list = os.listdir('.')
for elt in data_list:
    print(elt[:3]=='all')
    print(elt[3]!='_')

print([elt for elt in data_list if ('all'==elt[:3] and (elt[3]!='_'))])
input_hds = with_substr(data_list, '.hd')
input_pkls = with_substr(data_list, '.pkl')


[history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts] = ld_obj([elt for elt in data_list if ('all'==elt[:3] and (elt[3]!='_'))][0])

model = LR(x_train.shape[1], 2)


chckptNm = with_substr(input_hds, 'all')[0]

model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(chckptNm)

evl_tr = model.evaluate(x=x_train, y=y_train)
print('train: '+str(evl_tr))

evl_vl = model.evaluate(x=x_val, y=y_val)
print('val: '+str(evl_vl))

evl_ts = model.evaluate(x=x_tst, y=y_tst)
print('test: '+str(evl_ts))


w = 64
h = 3
nb_classes = 2
model = ResnetBuilder.build_resnet_18((h,w,w),nb_classes)
kr, ar, dr = 0.1, 0.01, 0.5
model = changeFinalLayer(model, nb_classes, kr, ar, dr)
model.compile(optimizer=optim, loss='mse', metrics=['accuracy'])

tps = ['T1', 'T2', 'demo']
for tp in tps:
    [history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts] = ld_obj(with_substr(input_pkls, tp)[0])
    if tp == 'demo':
        model = LR(x_train.shape[1], 1, reg=True, m=True)
        model.compile(optimizer=optim, loss='mse', metrics=['accuracy'])
    else:
        print(tp)
    model.load_weights(with_substr(input_hds, tp)[0])
    lasts = []
    labels = []
    
    for p,y,nm,x in zip([pred1_tr, pred1_vl, pred1_ts], [y_train, y_val, y_tst], ['tr','val','tst'], [x_train, x_val, x_tst]):
        evl_ts = model.evaluate(x=x, y=y)
        print(nm+': '+str(evl_ts))
        
        pred1_tr = model.predict(x)
        if tp != 'demo':
        
            intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[-4].output)#model.get_layer(layer_name).output
            int_pred1_tr = intermediate_layer_model.predict(x)
            lasts.append(int_pred1_tr)
            labels.append(y)
            print(int_pred1_tr.shape)
            
            ##
            '''
            
            predictions = model.predict(x)
            
            predicted_class = np.argmax(predictions)
            cam, heatmap = grad_cam(model, x, predictions, "block5_conv3")
            cv2.imwrite("gradcam.jpg", cam)
            
            register_gradient()
            guided_model = modify_backprop(model, 'GuidedBackProp')
            saliency_fn = compile_saliency_function(guided_model)
            saliency = saliency_fn([preprocessed_input, 0])
            gradcam = saliency[0] * heatmap[..., np.newaxis]
            cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
            '''
    if tp != 'demo':
        X_tsne = np.vstack(lasts)
        labels = np.hstack(labels)
        trtsvl = np.hstack([np.hstack(elt.shape[0]*[i]) for i,elt in enumerate(lasts)])
        
        
        #123 for trn, tst, val
        # 123 for label
        '''
        
        Y = tsne(X_tsne, 2, 50, 200.0)
        pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
        pylab.show()
        pylab.scatter(Y[:, 0], Y[:, 1], 20, labels>2)
        pylab.show()
        '''
        Y = tsne(X_tsne, 2, X_tsne.shape[1], 200.0)
        plt.scatter(Y[:, 0], Y[:, 1], 20, labels>2)
        plt.show()
        plt.savefig(tp)
        
        #preprocessed_input = load_image(sys.argv[1])





        '''
        
        sys.exit()
        compares = np.vstack([p.flatten(), np.array(y)]).transpose()
        tmp = compares>2.5
        print(nm+':'+str(np.mean(tmp[:,0] == tmp[:,1])))
        '''

[history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts] = ld_obj(with_substr(input_pkls, 'T1')[0])
[history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred2_tr, pred2_vl, pred2_ts] = ld_obj(with_substr(input_pkls, 'T2')[0])
[history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, predD_tr, predD_vl, predD_ts] = ld_obj(with_substr(input_pkls, 'demo')[0])


#####################
[history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts] = ld_obj([elt for elt in data_list if ('all'==elt[:3] and (elt[3]!='_'))][0])


def binarizeHiLo(l):
    return [elt>2 for elt in l]
nb_classes = 2

nb_epoch, batch_size = 100000, 32#100000
#predA_tr, predA_ts,predA_vl, collectA = runDemoLR_trn_tst_val(x_train.shape[1], nb_classes, batch_size, nb_epoch, x_train, binarizeHiLo(trnLs), x_tst, binarizeHiLo(tstLs), x_val, binarizeHiLo(valLs), fl_nm, tp_str = 'all', process = False)


dataA = []
repetition = 0
good = 0
while(1):
    predA_tr, predA_ts,predA_vl, collectA = runDemoLR_trn_tst_val(x_train.shape[1], nb_classes, batch_size, nb_epoch, x_train, binarizeHiLo(trnLs), x_tst, binarizeHiLo(tstLs), x_val, binarizeHiLo(valLs), fl_nm, tp_str = 'all'+str(repetition), process = False)
    repetition = repetition + 1
    [history, mn, s, x_train, y_train, x_val, y_val,x_tst, y_tst, pred1_tr, pred1_vl, pred1_ts, evl_tr, evl_vl, evl_ts] = collectA
    print([evl_tr, evl_vl, evl_ts])
    if np.all(np.array([evl_vl[1], evl_ts[1]])>0.80):
        good = good + 1
        dataA.append([predA_tr, predA_ts, predA_vl, evl_tr, evl_vl, evl_ts])
    if good > 10:
        break

allhdf5 = with_substr(input_hds, 'all')


input_pkl = with_substr(data_list, 'all_input_data')
tmp=ld_obj(input_pkl[0])
[trnIms, valIms, tstIms, trnLs, valLs, tstLs, trnDs, valDs, tstDs]=ld_obj(input_pkl[0])
#print(input_pkl)
Y = tsne(X, 2, 50, 20.0)
sys.exit()


[trnIms, valIms, tstIms, trnLs, valLs, tstLs, trnDs, valDs, tstDs]=ld_obj('../h5/' +sv_root+ 'all_input_data')



outPutFNm = '../output' + strftime("%H_%M_%S", gmtime()) + os.path.basename(__file__);
kr,ar=1,1
K.clear_session()

fl_nm = strftime("%H_%M_%S", gmtime()) + os.path.basename(__file__)
print(fl_nm)

sv_root = fl_nm[:-3] + '/'
os.mkdir('../h5/'+sv_root)




my_globs.initGLobals()
w = 112
h = 3
def getInps(spl = 0.3, Topt = np.array([1])):
    st = loadSt()
    Topt = np.array([1,2])
    inps, outs, demos = getFurmanReg(st, Topt)
    outs = outs.astype(float)
    ims, demos = compileImsDemos(inps, demos, w, h)
    rsz_inps = stack_obj_ar(ims)
    rsz_inps = [[im,d] for im, d in zip(rsz_inps, demos)]
    spl = [0.7, 0.1, 0.2]
    trnLs, trnIms, tstLs, tstIms, valLs, valIms = split_trn_tst_val(rsz_inps, outs, spl)
    print(outs)
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

##	weighted
weighted = False


nb_classes = len(np.unique(trnLs+tstLs+valLs))
print('Data loaded')

##############################
print('T1')

x_train, y_train = trnIms[:,:,:,:,0], trnLs
x_val, y_val = valIms[:,:,:,:,0], valLs
x_tst, y_tst = tstIms[:,:,:,:,0], tstLs

chknm = '../h5/T1'+fl_nm[:-2]+'h5'
learning_rate = 0.02
SGD_mom = 0.9
SGD_dec = 0

dr = 0.5
n = 32
seq_s = 'T1'
print('this vers saves as: ' + chknm)

model1 = ResnetBuilder.build_resnet_18((h,w,w),nb_classes)
model1 = changeFinalLayer(model1, 1, kr, ar, dr, reg=True, m=True)


pred1_tr, pred1_ts, pred1_vl, collect1 = seq_run_feat_cent_trn_tst_val(x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model1, weighted=weighted, sv_root = sv_root, loss_tp='mse')#, epoch_num=1
#print(np.vstack([pred1_tr, y_train]))
print(np.mean((pred1_tr>2.5) == (np.array(y_train)>2.5)))
print(np.mean((pred1_ts>2.5) == (np.array(y_tst)>2.5)))
print(np.mean((pred1_vl>2.5) == (np.array(y_val)>2.5)))
##############################
print('T2')

x_train, y_train = trnIms[:,:,:,:,1], trnLs
x_val, y_val = valIms[:,:,:,:,1], valLs
x_tst, y_tst = tstIms[:,:,:,:,1], tstLs

chknm = '../h5/T2'+fl_nm[:-2]+'h5'
learning_rate = 0.02
SGD_mom = 0.9
SGD_dec = 0


dr = 0.5
n = 32
seq_s = 'T2'
print('this vers saves as: ' + chknm)

model2 = ResnetBuilder.build_resnet_18((h,w,w),nb_classes)
model2 = changeFinalLayer(model2, 1, kr, ar, dr, reg=True, m=True)

pred2_tr, pred2_ts, pred2_vl, collect2 = seq_run_feat_cent_trn_tst_val(x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model2, weighted=weighted, sv_root = sv_root, loss_tp='mse')#, epoch_num=1

#print(np.vstack([pred1_tr, y_train]))
print(np.mean((pred1_tr>2.5) == (np.array(y_train)>2.5)))
print(np.mean((pred1_ts>2.5) == (np.array(y_tst)>2.5)))
print(np.mean((pred1_vl>2.5) == (np.array(y_val)>2.5)))
##############################
print('demo')
nb_epoch, batch_size = 100000, 32#100000
predD_tr, predD_ts,predD_vl, collectD = runDemoLR_trn_tst_val_reg(len(tstDs[0]), 1, batch_size, nb_epoch, trnDs, trnLs, tstDs, tstLs, valDs, valLs, fl_nm, weighted=weighted, sv_root = sv_root,loss_tp='mse', reg=True, m=True)


print(np.mean((predD_tr*4>2.5) == (np.array(y_train)*4>2.5)))
print(np.mean((predD_ts*4>2.5) == (np.array(y_tst)*4>2.5)))
print(np.mean((predD_vl*4>2.5) == (np.array(y_val)*4>2.5)))
##############################
print('all')
print(pred1_tr, pred2_tr, predD_tr)
x_train = np.hstack([pred1_tr, pred2_tr, predD_tr])
x_val = np.hstack([pred1_vl, pred2_vl, predD_vl])
x_tst = np.hstack([pred1_ts, pred2_ts, predD_ts])

print(x_train)

def binarizeHiLo(l):
    return [elt>2 for elt in l]
a=binarizeHiLo(trnLs)
nb_classes = 2

predA_tr, predA_ts,predA_vl, collectA = runDemoLR_trn_tst_val(x_train.shape[1], nb_classes, batch_size, nb_epoch, x_train, binarizeHiLo(trnLs), x_tst, binarizeHiLo(tstLs), x_val, binarizeHiLo(valLs), fl_nm, tp_str = 'all', process = False, weighted=weighted, sv_root = sv_root)
save_obj([trnIms, valIms, tstIms, trnLs, valLs, tstLs, trnDs, valDs, tstDs], '../h5/' +sv_root+ 'all_input_data')
K.clear_session()
#global toPrint
print(my_globs.toPrint)
print(fl_nm, learning_rate, kr, ar, dr )


with open(outPutFNm, "a") as file:
    file.write(str((kr,ar))+'\n') 
    file.write(my_globs.toPrint) 
    file.write(sv_root+'\n')
print(outPutFNm)