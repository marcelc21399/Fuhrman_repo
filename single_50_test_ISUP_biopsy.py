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
from utils import ld_obj, getFurmanReg, getMalign, save_obj, bothIms_and_Demo, runDemoLR_trn_tst_val,seq_run_feat_cent_trn_tst_val, split_trn_tst_val,perfMeasures, seq_run_feat_cent, stack_obj_ar, runDemoLR, getMdlInps, save_obj, parseDir, rsz, weighted_categorical_crossentropy, my_rot, rsz3D, split, plotSlices, getMalign, getClassWeights, hist_type, seq_run, getFurman, elastic_transform, single_im_batch_generator, LR, changeFinalLayer
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
	[trnLs, trnIms, tstLs, tstIms, valLs, valIms] = ld_obj('../inps_64_orthogonal')
	[biopsy_pats, ISUP_pats] = ld_obj('../misc_64_orthogonal')
	tstImsB, tstLsB = biopsy_pats[:, :4], biopsy_pats[:, 4] - 1
	tstImsI, tstLsI = ISUP_pats[:, :4], ISUP_pats[:, 4] - 1
	tstLsB, tstLsI = tstLsB.astype(bool), tstLsI.astype(bool)
	return trnLs, trnIms, tstLs, tstIms, valLs, valIms, tstLsI, tstImsI, tstLsB, tstImsB


trnLs, trnIms, tstLs, tstIms, valLs, valIms, tstLsI, tstImsI, tstLsB, tstImsB = getInps()

#ImsSet=tstIms
def resize_to_224_images(ImsSet):
	T1s = np.stack([np.stack([zoom(elt[:,:,sl], 3.5) for sl in range(3)], axis=2) for elt in ImsSet[:,0]], axis=0)
	T2s = np.stack([np.stack([zoom(elt[:,:,sl], 3.5) for sl in range(3)], axis=2) for elt in ImsSet[:,1]], axis=0)

	Ims = np.stack([T1s,T2s], axis = 4)

	demos = np.stack(ImsSet[:,2], axis=0)
	return Ims, demos

        
trnIms, trnDs = resize_to_224_images(trnIms)
valIms, valDs = resize_to_224_images(valIms)
tstIms, tstDs = resize_to_224_images(tstIms)
tstImsI, tstDsI = resize_to_224_images(tstImsI)
tstImsB, tstDsB = resize_to_224_images(tstImsB)

nb_classes = len(np.unique(np.hstack([trnLs,tstLs,valLs])))
print('Data loaded')


mn = np.mean(trnDs,axis=0)
std = np.std(trnDs,axis=0)
trnDs = (trnDs - mn)/std
tstDs = (tstDs - mn)/std
valDs = (valDs - mn)/std
tstDsI = (tstDsI - mn)/std
tstDsB = (tstDsB - mn)/std

##############################
T1_train, T2_train, y_train = trnIms[:,:,:,:,0],trnIms[:,:,:,:,1], np_utils.to_categorical(trnLs, nb_classes)
T1_val, T2_val, y_val = valIms[:,:,:,:,0], valIms[:,:,:,:,1], np_utils.to_categorical(valLs, nb_classes)
T1_tst, T2_tst, y_tst = tstIms[:,:,:,:,0], tstIms[:,:,:,:,1], np_utils.to_categorical(tstLs, nb_classes)
T1_tst_I, T2_tst_I, y_tst_I = tstImsI[:,:,:,:,0], tstImsI[:,:,:,:,1], np_utils.to_categorical(tstLsI, nb_classes)
T1_tst_B, T2_tst_B, y_tst_B = tstImsB[:,:,:,:,0], tstImsB[:,:,:,:,1], np_utils.to_categorical(tstLsB, nb_classes)


mn, s = -1,-1
def sampleWiseNorm(ims):
    return np.stack([(im-np.mean(im))/np.std(im) for im in ims],axis=0)
T1_train, T2_train, T1_val, T2_val, T1_tst, T2_tst, T1_tst_I, T2_tst_I, T1_tst_B, T2_tst_B = list(map(sampleWiseNorm, [T1_train, T2_train, T1_val, T2_val, T1_tst, T2_tst, T1_tst_I, T2_tst_I, T1_tst_B, T2_tst_B]))


#sys.exit()

x_train = [T1_train, T2_train, np.stack(trnDs,axis=0)]
x_val = [T1_val, T2_val, np.stack(valDs,axis=0)]
x_tst = [T1_tst, T2_tst, np.stack(tstDs,axis=0)]
x_tst_I = [T1_tst_I, T2_tst_I, np.stack(tstDsI,axis=0)]
x_tst_B = [T1_tst_B, T2_tst_B, np.stack(tstDsB,axis=0)]



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
			    
			weights = 1/hist_type(np.array(tstLs))
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



			evl_tr = model.evaluate(x=x_train, y=y_train)
			print('train: '+str(evl_tr))

			evl_val = model.evaluate(x=x_val, y=y_val)
			print('val: '+str(evl_val))

			evl_tst = model.evaluate(x=x_tst, y=y_tst)
			print('test: '+str(evl_tst))

			evl_tst_I = model.evaluate(x=x_tst_I, y=y_tst_I)
			print('testI: '+str(evl_tst_I))

			evl_tst_B = model.evaluate(x=x_tst_B, y=y_tst_B)
			print('testB: '+str(evl_tst_B))

			toPrint = toPrint + 'train: '+str(evl_tr) + '\n' + 'val: '+str(evl_val) + '\n' + 'test: '+str(evl_tst) + '\n' + 'testISUP: '+str(evl_tst_I) + '\n'+ 'testB: '+str(evl_tst_B) + '\n'






			pred1_tr = model.predict(x_train)
			pred1_vl = model.predict(x_val)
			pred1_ts = model.predict(x_tst)
			pred1_ts_I = model.predict(x_tst_I)

			pred1_ts_B = model.predict(x_tst_B)
			nm = fl_nm
			tp = 'multi'

			print(nm, tp)

			os.makedirs('../out/'+nm+tp)
			for p, y, trnvaltst in zip([pred1_tr, pred1_vl, pred1_ts, pred1_ts_I, pred1_ts_B],[y_train, y_val, y_tst, y_tst_I, y_tst_B],['trn','val','tst','tst_ISUP', 'tst_B']):
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