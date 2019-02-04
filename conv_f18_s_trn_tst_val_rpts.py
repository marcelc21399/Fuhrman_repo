#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:22:22 2018

@author: priscillachang
"""
import my_globs

from utils import ld_obj,save_obj, runDemoLR_trn_tst_val,seq_run_feat_cent_trn_tst_val, split_trn_tst_val,perfMeasures, seq_run_feat_cent, stack_obj_ar, runDemoLR, getMdlInps, save_obj, parseDir, rsz, weighted_categorical_crossentropy, my_rot, rsz3D, loadSt, split, plotSlices, getMalign, getClassWeights, hist_type, seq_run, register, getFurman, elastic_transform, single_im_batch_generator, compileImsDemos, LR, loadSt_orig, changeFinalLayer
import numpy as np

import os
from time import gmtime, strftime
import sys
sys.path.insert(0, 'keras-resnet-master')
from resnet_stoch_depth import ResnetBuilder


from keras import backend as K
from keras.utils import np_utils

outPutFNm = '../output' + strftime("%H_%M_%S", gmtime()) + os.path.basename(__file__);
P1s, P2s, PDs = [], [], []

for repetition in range(5):
	for kr in [0.05,0.07,0.1,0.15,0.2]:
		for ar in [0.007,0.01,0.013,0.016,0.02]:

			my_globs.initGLobals()


			K.clear_session()

			fl_nm = strftime("%H_%M_%S", gmtime()) + os.path.basename(__file__)
			print(fl_nm)

			sv_root = fl_nm[:-3] + '/'
			os.mkdir('../h5/'+sv_root)



			w = 64
			h = 3
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
			print(trnIms.shape, tstIms.shape)

			##	weighted
			weighted = False


			nb_classes = len(np.unique(trnLs+tstLs+valLs))
			print('Data loaded')

			##############################
			print('T1')

			x_train, y_train = trnIms[:,:,:,:,0], np_utils.to_categorical(trnLs, nb_classes)
			x_val, y_val = valIms[:,:,:,:,0], np_utils.to_categorical(valLs, nb_classes)
			x_tst, y_tst = tstIms[:,:,:,:,0], np_utils.to_categorical(tstLs, nb_classes)

			chknm = '../h5/T1'+fl_nm[:-2]+'h5'
			learning_rate = 0.2
			SGD_mom = 0.9
			SGD_dec = 0


			dr = 0.5
			n = 32
			seq_s = 'T1'
			print('this vers saves as: ' + chknm)

			model1 = ResnetBuilder.build_resnet_18((h,w,w),nb_classes)
			model1 = changeFinalLayer(model1, nb_classes, kr, ar, dr)
			model1.summary()


			pred1_tr, pred1_ts, pred1_vl, collect1 = seq_run_feat_cent_trn_tst_val(x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model1, weighted=weighted, sv_root = sv_root)
			P1s.append([pred1_tr, pred1_ts, pred1_vl])
			##############################
			print('T2')

			x_train, y_train = trnIms[:,:,:,:,1], np_utils.to_categorical(trnLs, nb_classes)
			x_val, y_val = valIms[:,:,:,:,1], np_utils.to_categorical(valLs, nb_classes)
			x_tst, y_tst = tstIms[:,:,:,:,1], np_utils.to_categorical(tstLs, nb_classes)

			chknm = '../h5/T2'+fl_nm[:-2]+'h5'
			learning_rate = 0.2
			SGD_mom = 0.9
			SGD_dec = 0

			dr = 0.5
			n = 32
			seq_s = 'T2'
			print('this vers saves as: ' + chknm)

			model2 = ResnetBuilder.build_resnet_18((h,w,w),nb_classes)
			model2 = changeFinalLayer(model2, nb_classes, kr, ar, dr)

			pred2_tr, pred2_ts, pred2_vl, collect2 = seq_run_feat_cent_trn_tst_val(x_train, y_train, x_tst, y_tst, x_val, y_val, chknm, learning_rate, SGD_mom, SGD_dec, kr, ar, dr, n, seq_s, fl_nm, h, w, tstLs, trnLs, valLs, model2, weighted=weighted, sv_root = sv_root)#, epoch_num=2
			P2s.append([pred2_tr, pred2_ts, pred2_vl])
			
			##############################
			print('demo')
			nb_epoch, batch_size = 100000, 32#100000
			predD_tr, predD_ts,predD_vl, collectD = runDemoLR_trn_tst_val(len(tstDs[0]), nb_classes, batch_size, nb_epoch, trnDs, trnLs, tstDs, tstLs, valDs, valLs, fl_nm, weighted=weighted, sv_root = sv_root)
			PDs.append([predD_tr, predD_ts,predD_vl])
			##############################
			print('all')
			x_train = np.transpose(np.vstack([elt[:,1] for elt in [pred1_tr, pred2_tr, predD_tr]]))
			x_val = np.transpose(np.vstack([elt[:,1] for elt in [pred1_vl, pred2_vl, predD_vl]]))
			x_tst = np.transpose(np.vstack([elt[:,1] for elt in [pred1_ts, pred2_ts, predD_ts]]))

			predA_tr, predA_ts,predA_vl, collectA = runDemoLR_trn_tst_val(x_train.shape[1], nb_classes, batch_size, nb_epoch, x_train, trnLs, x_tst, tstLs, x_val, valLs, fl_nm, tp_str = 'all', process = False, weighted=weighted, sv_root = sv_root)
			save_obj([trnIms, valIms, tstIms, trnLs, valLs, tstLs, trnDs, valDs, tstDs], '../h5/' +sv_root+ 'all_input_data')
			K.clear_session()
			#global toPrint
			print(kr,ar)
			print(my_globs.toPrint)

			with open(outPutFNm, "a") as file:
				file.write(str((kr,ar))+'\n') 
				file.write(my_globs.toPrint) 
				file.write(sv_root+'\n')
			print(outPutFNm)
			save_obj([P1s, P2s, PDs], '../h5/' +sv_root+ 'all_preds')
