#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:22:22 2018

@author: priscillachang
"""
import my_globs
import numpy as np

import os
from time import gmtime, strftime
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD
import sys

from keras import backend as K
from keras.utils import np_utils
from utils import ld_obj, getFurmanReg, getMalign, save_obj, bothIms_and_Demo, runDemoLR_trn_tst_val,seq_run_feat_cent_trn_tst_val, split_trn_tst_val,perfMeasures, seq_run_feat_cent, stack_obj_ar, runDemoLR, getMdlInps, save_obj, parseDir, rsz, weighted_categorical_crossentropy, my_rot, rsz3D, loadSt, split, plotSlices, getMalign, getClassWeights, hist_type, seq_run, register, getFurman, elastic_transform, single_im_batch_generator, compileImsDemos, LR, loadSt_orig, changeFinalLayer
import numpy as np
import csv
import os
from time import gmtime, strftime
import sys
sys.path.insert(0, 'keras-resnet-master')
from resnet_stoch_depth import ResnetBuilder

from scipy.ndimage import zoom
from matplotlib import pyplot as plt


from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from keras.applications.resnet50 import ResNet50




outPutFNm = '../output' + strftime("%H_%M_%S", gmtime()) + os.path.basename(__file__);

w = 64
h = 3
def getInps(spl = 0.3, Topt = np.array([1])):
    inps = ld_obj('../data/inps_RCC')
    ys = inps[0]+inps[2]
    xs = inps[1]+inps[3]
    
    xs_demo = np.stack(elt[1] for elt in xs)
        
    inps_all = ld_obj('../data/inps')
    
    new_inps = []
    for val in [1,3,5]:
        inds = []
        for elt in inps_all[val]:
            locn = np.where(np.all(elt[1] == xs_demo,axis=1))[0]
            if locn.size == 1:
                inds.append(locn[0])
        inds=np.array(inds)
        new_inps.append([bool(ys[elt]) for elt in inds])
        new_inps.append([xs[elt] for elt in inds])
        
    [trnLs, trnIms, tstLs, tstIms, valLs, valIms] = new_inps
    
    [tstImsF, tstLsF] = ld_obj('../data/inps_new_data_Fuhrman_cc.pkl')
    
    trnLs, trnIms = trnLs + tstLsF[:14], trnIms + tstImsF[:14]
    tstLs, tstIms = tstLs + tstLsF[14:28], tstIms + tstImsF[14:28]
    valLs, valIms = valLs + tstLsF[28:], valIms + tstImsF[28:]
    
    [tstImsI, tstLsI] = ld_obj('../data/inps_new_data_ISUP_cc.pkl')
    return trnLs, trnIms, tstLs, tstIms, valLs, valIms, tstLsI, tstImsI
'''
print(trnIms)
st=loadSt()
print(st[0])
sys.exit()
'''

trnLs, trnIms, tstLs, tstIms, valLs, valIms, tstLsI, tstImsI = getInps()

#ImsSet=tstIms
def resize_to_224_images(ImsSet):
    new = []
    for elt in ImsSet:
        ims, ds = elt
        new_ims = np.zeros((224,224,3,2))
        for seq in range(2):
            for slce in range(3):
                im = ims[:,:,slce,seq]
                new_im = zoom(im, 3.5)
                new_ims[:,:,slce,seq] = new_im
        new.append([new_ims, ds])
    return new

        
trnIms = resize_to_224_images(trnIms)
valIms = resize_to_224_images(valIms)
tstIms = resize_to_224_images(tstIms)
tstImsI = resize_to_224_images(tstImsI)
    

trnDs = [elt[1] for elt in trnIms]
trnIms = [elt[0] for elt in trnIms]

tstDs = [elt[1] for elt in tstIms]
tstIms = [elt[0] for elt in tstIms]

valDs = [elt[1] for elt in valIms]
valIms = [elt[0] for elt in valIms]

tstDsI = [elt[1] for elt in tstImsI]
tstImsI = [elt[0] for elt in tstImsI]

trnIms = np.stack(trnIms, axis = 0)
tstIms = np.stack(tstIms, axis = 0)
valIms = np.stack(valIms, axis = 0)
tstImsI = np.stack(tstImsI, axis = 0)

nb_classes = len(np.unique(trnLs+tstLs+valLs))
print('Data loaded')



trnDs = np.stack(trnDs,axis=0)
valDs = np.stack(valDs,axis=0)
tstDs = np.stack(tstDs,axis=0)
tstDsI = np.stack(tstDsI,axis=0)



mn = np.mean(trnDs,axis=0)
std = np.std(trnDs,axis=0)
trnDs = (trnDs - mn)/std
tstDs = (tstDs - mn)/std
valDs = (valDs - mn)/std
tstDsI = (tstDsI - mn)/std

##############################
T1_train, T2_train, y_train = trnIms[:,:,:,:,0],trnIms[:,:,:,:,1], np_utils.to_categorical(trnLs, nb_classes)
T1_val, T2_val, y_val = valIms[:,:,:,:,0], valIms[:,:,:,:,1], np_utils.to_categorical(valLs, nb_classes)
T1_tst, T2_tst, y_tst = tstIms[:,:,:,:,0], tstIms[:,:,:,:,1], np_utils.to_categorical(tstLs, nb_classes)
T1_tst_I, T2_tst_I, y_tst_I = tstImsI[:,:,:,:,0], tstImsI[:,:,:,:,1], np_utils.to_categorical(tstLsI, nb_classes)


mn, s = -1,-1
def sampleWiseNorm(ims):
    return np.stack([(im-np.mean(im))/np.std(im) for im in ims],axis=0)
T1_train, T2_train, T1_val, T2_val, T1_tst, T2_tst, T1_tst_I, T2_tst_I = list(map(sampleWiseNorm, [T1_train, T2_train, T1_val, T2_val, T1_tst, T2_tst, T1_tst_I, T2_tst_I]))


#sys.exit()



for rep in range(5):
	for kr in [0.05,0.07,0.1,0.15,0.2]:
		for ar in [0.007,0.01,0.013,0.016,0.02]:
			toPrint = ''

			my_globs.initGLobals()
			K.clear_session()

			fl_nm = strftime("%H_%M_%S", gmtime()) + os.path.basename(__file__)
			print(fl_nm)
			sv_root = fl_nm[:-3] + '/'
			os.mkdir('../h5/'+sv_root)


			data_gen_args = dict(featurewise_center=False,
			        featurewise_std_normalization=False,
			        samplewise_center=False,
			        samplewise_std_normalization=False,
			        rotation_range=360,
			        width_shift_range=0.1,
			        height_shift_range=0.1,
			        horizontal_flip=True,
			        vertical_flip=True,shear_range=10,zoom_range=0.05
			        )
			T1_datagen = ImageDataGenerator(**data_gen_args)
			T2_datagen = ImageDataGenerator(**data_gen_args)

			# Provide the same seed and keyword arguments to the fit and flow methods
			seed = 1
			T1_datagen.fit(T1_train, augment=True, seed=seed)
			T2_datagen.fit(T2_train, augment=True, seed=seed)
			#data = zip(T1_generator, T2_generator)

			def generate_generator_multiple():
			    T1_generator = T1_datagen.flow(T1_train, y_train, seed=seed)
			    T2_generator = T2_datagen.flow(T2_train, trnDs, seed=seed)
			    while True:
			        T1,lbl = next(T1_generator)
			        T2, demo = next(T2_generator)
			        yield [T1, T2, demo], lbl  #Yield both images and their mutual label
			        
			model1 = ResNet50()
			model2 = ResNet50()

			model = bothIms_and_Demo(model1, model2, dr = 0.5, nb_classes = 2, kr = kr, ar = ar, nd = 10)


			optim = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
			    
			weights = 1/hist_type(np.array(tstLs+trnLs))
			weights = weights/np.linalg.norm(weights)*np.sqrt(len(weights))
			model.compile(optimizer=optim, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])
			    
			chknm='../h5/' +sv_root+'mult'+strftime("%H_%M_%S", gmtime())
			callbacks = [
			        ModelCheckpoint(chknm, monitor='val_loss', save_best_only=True, verbose=0),
			        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=40, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
			        EarlyStopping(monitor='val_loss', min_delta=0, patience=300, verbose=1),
			    ]

			    
			history = model.fit_generator(
			    generate_generator_multiple(),
			    epochs=10000,#10000
			    steps_per_epoch=np.ceil(len(trnDs)/32),
			    validation_data=([T1_val, T2_val, np.stack(valDs,axis=0)], y_val),
			    verbose=2,
			    callbacks=callbacks
			    )        
			model.load_weights(chknm)
			x_train = [T1_train, T2_train, np.stack(trnDs,axis=0)]
			x_val = [T1_val, T2_val, np.stack(valDs,axis=0)]
			x_tst = [T1_tst, T2_tst, np.stack(tstDs,axis=0)]
			x_tst_I = [T1_tst_I, T2_tst_I, np.stack(tstDsI,axis=0)]
            
			    
			evl_tr = model.evaluate(x=x_train, y=y_train)
			print('train: '+str(evl_tr))

			evl_val = model.evaluate(x=x_val, y=y_val)
			print('val: '+str(evl_val))

			evl_tst = model.evaluate(x=x_tst, y=y_tst)
			print('test: '+str(evl_tst))

			evl_tst_I = model.evaluate(x=x_tst_I, y=y_tst_I)
			print('test: '+str(evl_tst_I))

			toPrint = toPrint + 'train: '+str(evl_tr) + '\n' + 'val: '+str(evl_val) + '\n' + 'test: '+str(evl_tst) + '\n' + 'testISUP: '+str(evl_tst_I) + '\n'






			pred1_tr = model.predict(x_train)
			pred1_vl = model.predict(x_val)
			pred1_ts = model.predict(x_tst)
			pred1_ts_I = model.predict(x_tst_I)
			nm = fl_nm
			tp = 'multi'

			print(nm, tp)

			os.makedirs('../out/'+nm+tp)
			for p, y, trnvaltst in zip([pred1_tr, pred1_vl, pred1_ts, pred1_ts_I],[y_train, y_val, y_tst, y_tst_I],['trn','val','tst','tst_ISUP']):
				dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
				with open('../out/'+nm+tp+'/'+trnvaltst+'dict.csv', 'w') as csv_file:
					writer = csv.writer(csv_file)
					for key, value in dic_one.items():
						print(key)
						print(value)
						toPrint = toPrint + str((key, value))
						writer.writerow([key, value])
					toPrint = toPrint + '\n'

			with open(outPutFNm, "a") as file:
				file.write(str((kr,ar))+'\n') 
				file.write(toPrint) 
				file.write(sv_root+'\n')

			save_obj(history.history, '../out/'+nm+tp+'/'+'history' )

			print(outPutFNm)

			plt.close('all')