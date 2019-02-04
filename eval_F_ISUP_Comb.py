import os
import my_globs

my_globs.initGLobals()

from utils import LR,initDic, ld_obj,perfIter,  bothIms_and_Demo, runDemoLR_trn_tst_val_reg, getFurmanReg, save_obj, runDemoLR_trn_tst_val,seq_run_feat_cent_trn_tst_val, split_trn_tst_val,perfMeasures, seq_run_feat_cent, stack_obj_ar, runDemoLR, getMdlInps, save_obj, parseDir, rsz, weighted_categorical_crossentropy, my_rot, rsz3D, loadSt, split, plotSlices, getMalign, getClassWeights, hist_type, seq_run, register, getFurman, elastic_transform, single_im_batch_generator, compileImsDemos, LR, loadSt_orig, changeFinalLayer
import numpy as np

import os
import csv
from time import gmtime, strftime
import sys
sys.path.insert(0, 'keras-resnet-master')
from resnet_stoch_depth import ResnetBuilder
from matplotlib import pyplot as plt
from keras.utils import np_utils

from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from scipy.ndimage import zoom
fl_nm = 'final_combined_model'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""




collect = []
root = '../eval_F_ISUP_Comb/'
locn = root+'mult'

#load model and data as beforew = 64
h = 3
def getInps(spl = 0.3, Topt = np.array([1])):
    [trnLs, trnIms, tstLs, tstIms, valLs, valIms] = ld_obj('../data/inps')
    [tstImsF, tstLsF] = ld_obj('../data/inps_new_data_Fuhrman.pkl')
    
    trnLs, trnIms = trnLs + tstLsF[:14], trnIms + tstImsF[:14]
    tstLs, tstIms = tstLs + tstLsF[14:28], tstIms + tstImsF[14:28]
    valLs, valIms = valLs + tstLsF[28:], valIms + tstImsF[28:]
    
    [tstImsI, tstLsI] = ld_obj('../data/inps_new_data_ISUP.pkl')
    return trnLs, trnIms, tstLs, tstIms, valLs, valIms, tstLsI, tstImsI

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

x_train = [T1_train, T2_train, np.stack(trnDs,axis=0)]
x_val = [T1_val, T2_val, np.stack(valDs,axis=0)]
x_tst = [T1_tst, T2_tst, np.stack(tstDs,axis=0)]
x_tst_I = [T1_tst_I, T2_tst_I, np.stack(tstDsI,axis=0)]


# load model


kr, ar = 0.01, 0.001
    
model1 = ResNet50()
model2 = ResNet50()

model = bothIms_and_Demo(model1, model2, dr = 0.5, nb_classes = 2, kr = kr, ar = ar, nd = 10)


optim = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

weights = 1/hist_type(np.array(tstLs+trnLs))
weights = weights/np.linalg.norm(weights)*np.sqrt(len(weights))
model.compile(optimizer=optim, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])


model.load_weights(locn)


evl_tr = model.evaluate(x=x_train, y=y_train)
print('train: '+str(evl_tr))

evl_val = model.evaluate(x=x_val, y=y_val)
print('val: '+str(evl_val))

evl_tst = model.evaluate(x=x_tst, y=y_tst)
print('test: '+str(evl_tst))

evl_tst_I = model.evaluate(x=x_tst_I, y=y_tst_I)
print('test: '+str(evl_tst_I))



pred1_tr = model.predict(x_train)
pred1_vl = model.predict(x_val)
pred1_ts = model.predict(x_tst)
pred1_ts_I = model.predict(x_tst_I)

nm = fl_nm
tp = 'multi'




print(nm, tp)

os.makedirs('../out/'+nm+tp)
dict_list = []
for p, y, trnvaltst in zip([pred1_tr, pred1_vl, pred1_ts, pred1_ts_I],[y_train, y_val, y_tst, y_tst_I],['trn','val','tst','tst_ISUP']):
	dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
	dict_list.append(dic_one)
	with open('../out/'+nm+tp+'/'+trnvaltst+'dict.csv', 'w') as csv_file:
		writer = csv.writer(csv_file)
		for key, value in dic_one.items():
			print(key)
			print(value)
			writer.writerow([key, value])
#dic_one?


#collect.append([d1, d2, dD, dA, locn1, locn2, locnD])
plt.close('all')




def modified_PR_ROC(nm, p1, lbl):
	#p1, n, lbl=pred1_ts, 100000, y_tst
	n = 100000
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
	ROC=plt.plot(Far,Tar)
	exp1=plt.scatter([1-0.6],[0.9090909090909091],color='r')
	exp2=plt.scatter([1-0.4],[0.8181818181818182],color='g')
	exp3=plt.scatter([1-0.5],[0.7272727272727273],color='b')
	exp4=plt.scatter([1-0.3],[0.7272727272727273],color='c')
	exp5=plt.scatter([1-0.5],[0.7272727272727273],color='m')
	plt.legend([ROC[0], exp1,exp2,exp3,exp4,exp5],['Model: Acc = 80.6%','Expert 1: Acc = 70.9%','Expert 2: Acc = 54.8%','Expert 3: Acc = 58.1%','Expert 4: Acc = 45.2%','Expert 5: Acc = 58.1%'])


	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.title("ROC Test Final Model")
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.savefig(nm+"ROC", format='eps', dpi=1000)
	AUCROC = -np.trapz(Tar, x=Far)

	Tar = np.array(PL)
	Far = np.array(RL)
	plt.figure()
	PR=plt.plot(Far,Tar)
	#exp1=plt.scatter([0.701],[0.466],color='r')
	#exp2=plt.scatter([0.628],[0.419],color='g')

	exp1=plt.scatter([0.9090909090909091],[0.5555555555555556],color='r')
	exp2=plt.scatter([0.8181818181818182],[0.42857142857142855],color='g')
	exp3=plt.scatter([0.7272727272727273],[0.4444444444444444],color='b')
	exp4=plt.scatter([0.7272727272727273],[0.36363636363636365],color='c')
	exp5=plt.scatter([0.7272727272727273],[0.4444444444444444],color='m')
	plt.legend([PR[0], exp1,exp2,exp3,exp4,exp5],['Model: Acc = 80.6%','Expert 1: Acc = 70.9%','Expert 2: Acc = 54.8%','Expert 3: Acc = 58.1%','Expert 4: Acc = 45.2%','Expert 5: Acc = 58.1%'])


	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title("Precision-Recall Test Final Model")
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.savefig(nm+"Precision", format='eps', dpi=1000)
	AUCPR = -np.trapz(Tar, x=Far)

	dic_act = initDic()
	dic_act = perfIter(p1, 0.5, dic_act, y1, yz)

	dic_act['AUCPR'] = AUCPR
	dic_act['AUCROC'] = AUCROC

modified_PR_ROC(fl_nm+'_F', pred1_ts, y_tst)
modified_PR_ROC(fl_nm+'_I', pred1_ts_I, y_tst_I)
    
    
def rdDic(d):
    print(d['AUCROC'],d['Acc'][0],1-d['FPR'][0],d['TPR'][0])

#tst_F
rdDic(dict_list[2])
#tst_I
rdDic(dict_list[3])