import os
import my_globs

my_globs.initGLobals()

from utils import LR,initDic, ld_obj,perfIter,  runDemoLR_trn_tst_val_reg, getFurmanReg, save_obj, runDemoLR_trn_tst_val,seq_run_feat_cent_trn_tst_val, split_trn_tst_val,perfMeasures, seq_run_feat_cent, stack_obj_ar, runDemoLR, getMdlInps, save_obj, parseDir, rsz, weighted_categorical_crossentropy, my_rot, rsz3D, loadSt, split, plotSlices, getMalign, getClassWeights, hist_type, seq_run, register, getFurman, elastic_transform, single_im_batch_generator, compileImsDemos, LR, loadSt_orig, changeFinalLayer
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

fl_nm = strftime("%H_%M_%S", gmtime()) + os.path.basename(__file__)




collect = []
root = '../Bag_Prob/'
locn1, locn2, locnD, locnA = root+'chckptT1.h5', root+'chckptT2.h5', root+'chckptdemo.hdf5', root+'chckptall.hdf5'
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

####################
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

pred1_tr = model.predict(x_train)
pred1_ts = model.predict(x_tst)
pred1_vl = model.predict(x_val)



def perfMetr(nm, tp, p_tr, p_ts, p_vl):
    #nm, tp, p_tr, p_ts, p_vl = nm, tp, predA_tr, predA_ts, predA_vl
    fol = '../out/'+nm+tp
    if not os.path.exists(fol):
        os.makedirs(fol)
    for p, y, trnvaltst in zip([p_tr, p_vl, p_ts],[y_train, y_val, y_tst],['trn','val','tst']):
        dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
        with open('../out/'+nm+tp+'/'+trnvaltst+'dict.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dic_one.items():
                print(key, value)
                my_globs.toPrint = my_globs.toPrint + str((key, value))
                writer.writerow([key, value])
    return dic_one

nm = 'Bag_Prob'
tp = 'T1'

d1 = perfMetr(nm, tp, pred1_tr, pred1_ts, pred1_vl)

########

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

pred2_tr = model.predict(x_train)
pred2_ts = model.predict(x_tst)
pred2_vl = model.predict(x_val)

nm = 'Bag_Prob'
tp = 'T2'

d2 = perfMetr(nm, tp, pred2_tr, pred2_ts, pred2_vl)
####################
trnDs = np.stack(trnDs, axis=0)
tstDs = np.stack(tstDs, axis=0)
valDs = np.stack(valDs, axis=0)

x_train, y_train = trnDs, np_utils.to_categorical(trnLs, nb_classes)
x_tst, y_tst = tstDs, np_utils.to_categorical(tstLs, nb_classes)
x_val, y_val = valDs, np_utils.to_categorical(valLs, nb_classes)

mn = np.mean(x_train, axis = 0)
s = np.std(x_train, axis = 0)
x_train = (x_train - mn)/s
x_val = (x_val - mn)/s
x_tst = (x_tst - mn)/s


nb_inp = x_train.shape[1]
nb_classes = 2
model = LR(nb_inp, nb_classes)

optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(locnD)

model.evaluate(x=x_train, y=y_train)

model.evaluate(x=x_val, y=y_val)

model.evaluate(x=x_tst, y=y_tst)

predD_tr = model.predict(x_train)
predD_ts = model.predict(x_tst)
predD_vl = model.predict(x_val)

x_train = np.hstack([pred1_tr, pred2_tr, predD_tr])
x_val = np.hstack([pred1_vl, pred2_vl, predD_vl])
x_tst = np.hstack([pred1_ts, pred2_ts, predD_ts])


nm = 'Bag_Prob'
tp = 'Demo'

dD = perfMetr(nm, tp, predD_tr, predD_ts, predD_vl)
####################


nb_inp = x_train.shape[1]
nb_classes = 2
model = LR(nb_inp, nb_classes)


optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])


model.load_weights(locnA)

trVal=model.evaluate(x=x_train, y=y_train)
vlVal=model.evaluate(x=x_val, y=y_val)
tsVal=model.evaluate(x=x_tst, y=y_tst)


predA_tr = model.predict(x_train)
predA_ts = model.predict(x_tst)
predA_vl = model.predict(x_val)

print(trVal, vlVal, tsVal)

nm = 'Bag_Prob'
tp = 'All'

dA = perfMetr(nm, tp, predA_tr, predA_ts, predA_vl)

collect.append([d1, d2, dD, dA, locn1, locn2, locnD])
plt.close('all')





p1, n, lbl=predA_ts, 100000, y_tst
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

    
    
    
def rdDic(d):
    print(d['AUCROC'],d['Acc'][0],1-d['FPR'][0],d['TPR'][0])

rdDic(d1)
rdDic(d2)
rdDic(dD)
rdDic(dA)