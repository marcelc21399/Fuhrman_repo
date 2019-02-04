import os
import my_globs

my_globs.initGLobals()

from utils import initDic, perfIter, LR, ld_obj,perfIter,  runDemoLR_trn_tst_val_reg, getFurmanReg, save_obj, runDemoLR_trn_tst_val,seq_run_feat_cent_trn_tst_val, split_trn_tst_val,perfMeasures, seq_run_feat_cent, stack_obj_ar, runDemoLR, getMdlInps, save_obj, parseDir, rsz, weighted_categorical_crossentropy, my_rot, rsz3D, loadSt, split, plotSlices, getMalign, getClassWeights, hist_type, seq_run, register, getFurman, elastic_transform, single_im_batch_generator, compileImsDemos, LR, loadSt_orig, changeFinalLayer
import numpy as np

import os
import csv
from time import gmtime, strftime
from keras.models import Model
import sys
from tsne import tsne
sys.path.insert(0, 'keras-resnet-master')
from resnet_stoch_depth import ResnetBuilder
from matplotlib import pyplot as plt
from keras.utils import np_utils

from keras.optimizers import SGD

toPrint2 = []

def binarizeHiLo(l):
    return [elt>2 for elt in l]

def acc(p, y):
    return np.mean(
            np.equal(p.flatten().astype(float)>2.5, np.array(y)>2)
            )
def rdDic(d):
    print(d['AUCROC'],d['Acc'][0],1-d['FPR'][0],d['TPR'][0])
    return str([d['AUCROC'],d['Acc'][0],1-d['FPR'][0],d['TPR'][0]])

collect = []
root = '../Bag_Reg/'
locn1, locn2, locnD, locnA = root+'chckptT1.hdf5', root+'chckptT2.hdf5', root+'chckptdemo.hdf5', root+'chckptall.hdf5'
w = 112
h = 3
nb_classes = 2
model = ResnetBuilder.build_resnet_18((h,w,w),nb_classes)
kr, ar, dr = 0.1, 0.01, 0.5
model = changeFinalLayer(model, 1, kr, ar, dr, reg=True, m=True)
optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optim, loss='mse', metrics=['accuracy'])
model.load_weights(locn1)
model.summary()



def with_substr(ls, sub_str):
    return [elt for elt in ls if sub_str in elt]

data_list = os.listdir('.')

input_hds = with_substr(data_list, '.hd')
input_pkls = with_substr(data_list, '.pkl')

####################

[trnIms, valIms, tstIms, trnLs, valLs, tstLs, trnDs, valDs, tstDs]=ld_obj(root + 'all_input_data')
#trnLs, trnIms, tstLs, tstIms, valLs, valIms = getInps()

x_train, y_train = trnIms[:,:,:,:,0], trnLs
x_val, y_val = valIms[:,:,:,:,0], valLs
x_tst, y_tst = tstIms[:,:,:,:,0], tstLs

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

trnAcc = acc(pred1_tr, y_train)
valAcc = acc(pred1_vl, y_val)
tstAcc = acc(pred1_ts, y_tst)
print(trnAcc, valAcc, tstAcc)

collect.append([trnAcc, valAcc, tstAcc, evl_tr, evl_vl, evl_ts])


def perfMetr(nm, tp, p_tr, p_ts, p_vl):
    #nm, tp, p_tr, p_ts, p_vl = nm, tp, predA_tr, predA_ts, predA_vl
    fol = '../out/'+nm+tp
    if not os.path.exists(fol):
        os.makedirs(fol)
    for p, y, trnvaltst in zip([p_tr, p_vl, p_ts],[y_train, y_val, y_tst],['trn','val','tst']):
        dic_all, dic_one = perfMeasures(p, 1000, y, nm = nm+tp+'/'+trnvaltst)
        my_globs.toPrint=my_globs.toPrint+rdDic(dic_one)
        toPrint2.append(str(dic_one))
    return dic_one

nm = 'Bag_Reg'
tp = 'T1'

intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-4].output)#model.get_layer(layer_name).output
X_tsne = intermediate_layer_model.predict(np.vstack([x_train, x_val, x_tst]))
truth = np.array(trnLs+valLs+tstLs)>2
Y = tsne(X_tsne, 2, X_tsne.shape[1], 200.0)
        
plt.figure()
F = truth != 1
T = truth == 1
L = plt.scatter(Y[F, 0], Y[F, 1], 20, 'g')
H = plt.scatter(Y[T, 0], Y[T, 1], 20, 'r')
plt.legend((L,H),('Low Fuhrman', 'High Fuhrman'))
plt.xlabel('t-SNE Axis 1')
plt.ylabel('t-SNE Axis 2')
plt.title('T1 Fuhrman Grade Neural Net Final Layer t-SNE')
#plt.show()
plt.savefig('T1_Bag_Reg_tSNE', format='eps', dpi = 1000)

########

model.load_weights(locn2)

x_train, y_train = trnIms[:,:,:,:,1], trnLs
x_val, y_val = valIms[:,:,:,:,1], valLs
x_tst, y_tst = tstIms[:,:,:,:,1], tstLs

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

nm = 'Bag_Reg'
tp = 'T2'

trnAcc = acc(pred2_tr, y_train)
valAcc = acc(pred2_vl, y_val)
tstAcc = acc(pred2_ts, y_tst)
print(trnAcc, valAcc, tstAcc)
collect.append([trnAcc, valAcc, tstAcc, evl_tr, evl_vl, evl_ts])









intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-4].output)#model.get_layer(layer_name).output
X_tsne = intermediate_layer_model.predict(np.vstack([x_train, x_val, x_tst]))
truth = np.array(trnLs+valLs+tstLs)>2
Y = tsne(X_tsne, 2, X_tsne.shape[1], 200.0)
        
plt.figure()
F = truth != 1
T = truth == 1
L = plt.scatter(Y[F, 0], Y[F, 1], 20, 'g')
H = plt.scatter(Y[T, 0], Y[T, 1], 20, 'r')
plt.legend((L,H),('Low Fuhrman', 'High Fuhrman'))
plt.xlabel('t-SNE Axis 1')
plt.ylabel('t-SNE Axis 2')
plt.title('T2 Fuhrman Grade Neural Net Final Layer t-SNE')
#plt.show()
plt.savefig('T2_Bag_Reg_tSNE', format='eps', dpi = 1000)


        
        

#sys.exit()

####################
trnDs = np.stack(trnDs, axis=0)
tstDs = np.stack(tstDs, axis=0)
valDs = np.stack(valDs, axis=0)

x_train, y_train = trnDs, trnLs
x_tst, y_tst = tstDs, tstLs
x_val, y_val = valDs, valLs

mn = np.mean(x_train, axis = 0)
s = np.std(x_train, axis = 0)
x_train = (x_train - mn)/s
x_val = (x_val - mn)/s
x_tst = (x_tst - mn)/s


nb_inp = x_train.shape[1]
nb_classes = 1
model = LR(x_train.shape[1], 1, reg=True, m=True)

optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optim, loss='mse', metrics=['accuracy'])
model.load_weights(locnD)

evl_tr = model.evaluate(x=x_train, y=y_train)
print('train: '+str(evl_tr))

evl_vl = model.evaluate(x=x_val, y=y_val)
print('val: '+str(evl_vl))

evl_ts = model.evaluate(x=x_tst, y=y_tst)
print('test: '+str(evl_ts))

predD_tr = model.predict(x_train)
predD_ts = model.predict(x_tst)
predD_vl = model.predict(x_val)

trnAcc = acc(predD_tr, y_train)
valAcc = acc(predD_vl, y_val)
tstAcc = acc(predD_ts, y_tst)
print(trnAcc, valAcc, tstAcc)

collect.append([trnAcc, valAcc, tstAcc, evl_tr, evl_vl, evl_ts])
x_train = np.hstack([pred1_tr, pred2_tr, predD_tr])
x_val = np.hstack([pred1_vl, pred2_vl, predD_vl])
x_tst = np.hstack([pred1_ts, pred2_ts, predD_ts])


nm = 'Bag_Reg'
tp = 'Demo'
####################
nb_classes = 2

y_train = np_utils.to_categorical(binarizeHiLo(trnLs), nb_classes)
y_tst = np_utils.to_categorical(binarizeHiLo(tstLs), nb_classes)
y_val = np_utils.to_categorical(binarizeHiLo(valLs), nb_classes)


nb_inp = x_train.shape[1]

model = LR(nb_inp, nb_classes)


optim = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])


model.load_weights(locnA)

evl_tr = model.evaluate(x=x_train, y=y_train)
print('train: '+str(evl_tr))

evl_vl = model.evaluate(x=x_val, y=y_val)
print('val: '+str(evl_vl))

evl_ts = model.evaluate(x=x_tst, y=y_tst)
print('test: '+str(evl_ts))


predA_tr = model.predict(x_train)
predA_ts = model.predict(x_tst)
predA_vl = model.predict(x_val)

nm = 'Bag_Reg'
tp = 'All'

dA = perfMetr(nm, tp, predA_tr, predA_ts, predA_vl)

plt.close('all')





p1, n, lbl=predA_vl, 100000, y_val
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
plt.legend([ROC[0], exp1,exp2,exp3,exp4,exp5],['Model: Acc = 81.8%','Expert 1: Acc = 70.9%','Expert 2: Acc = 54.8%','Expert 3: Acc = 58.1%','Expert 4: Acc = 45.2%','Expert 5: Acc = 58.1%'])



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
exp1=plt.scatter([0.9090909090909091],[0.5555555555555556],color='r')
exp2=plt.scatter([0.8181818181818182],[0.42857142857142855],color='g')
exp3=plt.scatter([0.7272727272727273],[0.4444444444444444],color='b')
exp4=plt.scatter([0.7272727272727273],[0.36363636363636365],color='c')
exp5=plt.scatter([0.7272727272727273],[0.4444444444444444],color='m')
plt.legend([PR[0], exp1,exp2,exp3,exp4,exp5],['Model: Acc = 81.8%','Expert 1: Acc = 70.9%','Expert 2: Acc = 54.8%','Expert 3: Acc = 58.1%','Expert 4: Acc = 45.2%','Expert 5: Acc = 58.1%'])


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

    
    
    

for elt in collect:
    print(elt[2], elt[5])



rdDic(d1)
rdDic(d2)
rdDic(dD)
rdDic(dA)

print(111)
print(my_globs.toPrint)
for d in toPrint2:
    print(d)
for d in toPrint2:
    rdDic(d)
