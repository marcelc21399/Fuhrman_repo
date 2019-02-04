#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:22:22 2018

@author: priscillachang
"""
import os
'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
'''
import my_globs
my_globs.initGLobals()

from utils import runDemoLR_trn_tst_val_reg, getFurmanReg, save_obj, runDemoLR_trn_tst_val,seq_run_feat_cent_trn_tst_val, split_trn_tst_val,perfMeasures, seq_run_feat_cent, stack_obj_ar, runDemoLR, getMdlInps, save_obj, parseDir, rsz, weighted_categorical_crossentropy, my_rot, rsz3D, loadSt, split, plotSlices, getMalign, getClassWeights, hist_type, seq_run, register, getFurman, elastic_transform, single_im_batch_generator, compileImsDemos, LR, loadSt_orig, changeFinalLayer
import numpy as np

import os
from time import gmtime, strftime
import sys
sys.path.insert(0, 'keras-resnet-master')
from resnet_stoch_depth import ResnetBuilder

from keras import backend as K
from keras.utils import np_utils

outPutFNm = '../output' + strftime("%H_%M_%S", gmtime()) + os.path.basename(__file__);

for rpt_num in range(4):
    for kr in [0.05,0.07,0.1,0.15,0.2]:
        for ar in [0.007,0.01,0.013,0.016,0.02]:
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


            dr=0.5
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