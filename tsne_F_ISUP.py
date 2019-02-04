import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

plt.close("all")


import my_globs

from tsne import tsne
from keras.models import Model

my_globs.initGLobals()

from utils import LR,initDic, ld_obj,perfIter,  bothIms_and_Demo, runDemoLR_trn_tst_val_reg, getFurmanReg, save_obj, runDemoLR_trn_tst_val,seq_run_feat_cent_trn_tst_val, split_trn_tst_val,perfMeasures, seq_run_feat_cent, stack_obj_ar, runDemoLR, getMdlInps, save_obj, parseDir, rsz, weighted_categorical_crossentropy, my_rot, rsz3D, loadSt, split, plotSlices, getMalign, getClassWeights, hist_type, seq_run, register, getFurman, elastic_transform, single_im_batch_generator, compileImsDemos, LR, loadSt_orig, changeFinalLayer
import numpy as np

import os
import csv
from time import gmtime, strftime
import sys
sys.path.insert(0, 'keras-resnet-master')
from resnet_stoch_depth import ResnetBuilder


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

model.summary()

nm = fl_nm
tp = 'multi'



lasts = []
labels = []
truth = []

intermediate_layer_model = Model(inputs=model.input,
                         outputs=model.layers[-2].output)#model.get_layer(layer_name).output

for y,nm,x in zip([y_train, y_val, y_tst, y_tst_I], ['tr','val','tst_F','tst_F'], [x_train, x_val, x_tst, x_tst_I]):
	evl = model.evaluate(x=x, y=y)
	print(nm+': '+str(evl))

	pred1_tr = model.predict(x)
	print(pred1_tr)
	lbls = np.argmax(pred1_tr, axis = 1)
	print(lbls)
	labels.append(lbls)

	truth.append(np.argmax(y, axis = 1))


	int_pred1_tr = intermediate_layer_model.predict(x)
	lasts.append(int_pred1_tr)
	print(int_pred1_tr.shape)


    

X_tsne = np.vstack(lasts)
trtsvl = np.hstack([np.hstack(elt.shape[0]*[i]) for i,elt in enumerate(lasts)])
labels = np.hstack(labels)
truth = np.hstack(truth)
print(truth)

szs = list(map(len,[y_train, y_val, y_tst, y_tst_I]))
inds = []
first = 0
for elt in szs:
    tmp = np.zeros((sum(szs),))
    tmp[first:first+elt] = 1
    first = first+elt
    inds.append(tmp)
    

#123 for trn, tst, val
# 123 for label
'''

Y = tsne(X_tsne, 2, 50, 200.0)
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.show()
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels>2)
pylab.show()
'''

Y = tsne(X_tsne, 2, 50, 100)
#plt.figure()
plt.close('all')

plt.scatter(Y[:, 0], Y[:, 1], 20, truth)
plt.savefig('combined_tsne', format='eps', dpi = 100)



#plt.figure()
plt.close('all')

F = truth != 1
T = truth == 1
L = plt.scatter(Y[F, 0], Y[F, 1], 20, 'g')
H = plt.scatter(Y[T, 0], Y[T, 1], 20, 'r')
plt.legend((L,H),('Low Fuhrman', 'High Fuhrman'))
plt.xlabel('t-SNE Axis 1')
plt.ylabel('t-SNE Axis 2')
plt.title('Combined Fuhrman/ISUP Grade Neural Net Final Layer t-SNE')
#plt.show()
plt.savefig('combined_tsne_labelled._High_Low', format='eps', dpi = 1000)

#plt.figure()
plt.close('all')


indices = np.logical_and(F, inds[0])
trg = plt.scatter(Y[indices, 0], Y[indices, 1], 20, 'g', marker="v")
indices = np.logical_and(T, inds[0])
trr = plt.scatter(Y[indices, 0], Y[indices, 1], 20, 'r', marker="v")

indices = np.logical_and(F, inds[1])
vlg = plt.scatter(Y[indices, 0], Y[indices, 1], 20, 'g', marker="^")
indices = np.logical_and(T, inds[1])
vlr = plt.scatter(Y[indices, 0], Y[indices, 1], 20, 'r', marker="^")

indices = np.logical_and(F, inds[2])
tsg = plt.scatter(Y[indices, 0], Y[indices, 1], 20, 'g', marker="<")
indices = np.logical_and(T, inds[2])
tsr = plt.scatter(Y[indices, 0], Y[indices, 1], 20, 'r', marker="<")

indices = np.logical_and(F, inds[3])
tsig = plt.scatter(Y[indices, 0], Y[indices, 1], 20, 'g', marker=">")
indices = np.logical_and(T, inds[3])
tsir = plt.scatter(Y[indices, 0], Y[indices, 1], 20, 'r', marker=">")


#plt.legend((L,H),('Low Fuhrman', 'High Fuhrman'))
plt.xlabel('t-SNE Axis 1')
plt.ylabel('t-SNE Axis 2')
plt.title('Combined Fuhrman/ISUP Grade Neural Net Final Layer t-SNE')
#plt.show()
plt.savefig('combined_tsne_labelled_trn_tst', format='eps', dpi = 1000)